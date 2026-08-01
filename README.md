# inklingrs

An inference engine for [Inkling-Small](https://thinkingmachines.ai/news/inkling-small/)
on Apple Silicon, in Rust + Metal. The two things it aims to do that existing
runtimes do not: **multi-token prediction** and **continuous batching**.

## Layout

    crates/inkling-core     config, checkpoint loading, architecture
    crates/inkling-metal    Metal backend; kernels compiled at runtime
    crates/inkling-cli      binary
    reference/              Python mlx-vlm oracle, kept out of the Rust tree
    models/                 weights (gitignored)

`inkling-serve` splits out of `inkling-cli` once the batching scheduler is more
than a request loop.

## Getting started

    direnv allow          # or: nix develop
    just sync             # reference venv + mlx-vlm patches
    just test

### Which of the three test runs to use

`just test` is the one to run while iterating: **467 of the 476 tests, no
checkpoint, ten seconds.** Everything a fixture can settle is here — the
kernels against the CPU, the CPU against mlx-vlm's recorded activations, the
tokenizer against the whole vocabulary, the server against its own frames. The
30 that need weights report a skip and pass. It runs through libtest, which puts
a crate's tests in one process: opening a Metal device costs a second, so the 133
kernel tests are 7.5 s sharing a process and 161 s with one each. Nothing in this
tier measures the process it runs in, which is what makes sharing one free.

`just test-full` is what has to pass before a commit lands: **all 476 against a
real checkpoint, four minutes thirty.** The 36 gated tests are what
only weights can settle — that the packed tensors decode to what the reference
decodes, that 42 trained layers reproduce the recorded stack, that the engine
generates the oracle's own continuation — and `--backend cpu` is the oracle they
are measured against, at 9.0 s a decoded token, which is where most of those
minutes go. This tier runs a process a test, which is what keeps a test that
bounds its resident set bounding only its own.

`just test-timing` is the nine tests whose result *is* a number — a duration
they assert on, a resident set they bound, the two decode-step tables quoted
above — run one at a time with nothing beside them. **A measurement taken while
nine other tests ran is a measurement of the nine:** a round trip this repo has at
191 µs reports 598 under a parallel suite, and `.config/nextest.toml` records
what believing a number like that once cost. `#[ignore]` is what keeps them out
of the two runs above, and what selects them here.

Text in, text out, streamed to stdout as each token is decoded:

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' -n 4

A decode step is about 34 ms against mlx-vlm's 32 ms, and the timings go to
stderr so stdout stays pipeable. The prompt reaches the tokenizer as it stands,
so the model *continues* it rather than answering it. A chat turn is written out
in full — `<|message_user|><|content_text|>…<|end_message|><|message_model|>` —
rather than applied by a template this does not implement.

**Every matmul in the model runs on the GPU, and no weight one of them reads is
ever decoded to memory** — the MXFP4 ones in registers a nibble at a time, the
routers' bfloat16 gates by a shift — and `--backend cpu` puts them all back:
0.047 s a token against the CPU's 9.0. The experts were the first two thirds of
that. A token reads 6 of each MoE layer's 256 experts and both of its shared
ones, which is 32 GB of float32 the CPU path decodes to multiply against and 4.3
GB of packed bytes the GPU path indexes into and never decodes at all. The rest
is every layer's own projections — five for attention, three more on each of the
two dense layers — which are 9 GB of float32 that *every* token reads all of.

**Nothing is copied onto the device.** The forty layers' banks are 137 GB, which
is the whole checkpoint but for its two ends, and they are handed to the GPU
where the checkpoint mapped them — `newBufferWithBytesNoCopy` over all of it in
6 ms. So the resident set goes *down* — 20.8 GiB with only the head there, 2.4
GiB once the banks are, and 0.12 GiB once the layers' own projections, norms and
gates are too — and a bank nobody routes to costs nothing to have wrapped. Note
what those numbers stop meaning: the pages are still in the unified buffer
cache, they are simply no longer this process's.

The one thing here that is allocated rather than mapped is the keys and values,
and it is the only part of the footprint that grows with the sequence: each
layer keeps its own `[kv_heads, capacity, head_dim]` span of each and doubles it
when a sequence outruns it, which at 64 slots is 21 MB across the stack and puts
an eight-token generation at 0.14 GiB.

**What a step costs is now mostly the asking, and that is measured rather than
inferred.** Every operation a forward pass runs opens a scope charged the time
inside it that no scope inside *it* claimed, so the rows of a decode step sum to
the step and what they leave over is a number rather than a shrug:

    submit and wait       2    83%      of which the device executed for 26 ms
    dispatch encode    1077    14%
    readback              2     0%
    everything else                     3%

**Every row that named an operation of a layer has left it, and the shape of
what is left has not changed.** The routers' gates were 19% and every layer's
bfloat16 tensors widened again on every step were 8%; the first is a dispatch now
and the second happens once, at load. The attention step and the mask it added
were 1.4% between them, and the two are one dispatch now. The two short
convolutions inside attention and the two head norms beside them were 0.8% by
the rows that named them, and they are four more dispatches — what they cost was
never their own time. The last four to go were the router's softmax, the two
scatters that weighted what its banks answered, the add between those halves and
the layer's own second convolution: 0.8 ms between them, and 1.0 ms more of
readback behind them, because what a layer answers with is now one tensor rather
than five. **They cost more where they went than they cost here**, which is the
first handover in this project of which that is true — see the layer's own
paragraph below. Four fifths of a step is still a round trip, and the device is
now executing for **92%** of it — two submissions, around work that is no longer
waited for a layer at a time. Nothing an operation of a layer would open a scope
around is left in the table at all: what remains beside the round trip is
encoding it, the sampling at the end, and the embedding at the start.

**And now which kernel owns which of those 26 milliseconds.** The device
timestamps a command buffer, and a decode step is two of them around 1077
dispatches, so until this landed the 26 ms was one number with nine kernels
behind it. It is now nine numbers, each beside the bytes that dispatch said it
moves and what that comes to against this machine's 819 GB/s:

    kernel            calls    device   share       moved   achieved  of peak
    packed_matmul       457   21.08ms   71.7%   5932.36 MB   281 GB/s     34%
    rms_norm            168    2.53ms    8.6%      5.89 MB     2 GB/s      0%
    dense_matmul         40    2.08ms    7.1%     85.24 MB    41 GB/s      5%
    short_conv          168    1.30ms    4.4%     22.02 MB    17 GB/s      2%
    fused_attention      42     763µs    2.6%      5.62 MB     7 GB/s      1%
    router_top_k         40     545µs    1.9%      0.08 MB     0 GB/s      0%
    swiglu               82     395µs    1.3%      8.26 MB    21 GB/s      3%
    router_weights       40     377µs    1.3%      0.00 MB     0 GB/s      0%
    moe_combine          40     322µs    1.1%      5.90 MB    18 GB/s      2%

**The packed matmul is 72% of the device's time and it is the only kernel here
doing bandwidth's work.** Its 5.9 GB is what the checkpoint's shapes say a token
reads — six of each MoE layer's 256 experts and both shared ones, plus every
layer's own projections — arrived at from a dispatch's own declaration rather
than from that arithmetic, and the two agree. 34% of the machine is not a
finished kernel and it is not a decade off one either; M2's isolated matmul
measured 284 to 424 GB/s, so 281 is the bottom of the range this kernel has
already been seen to reach.

**The surprise is the other 28%.** The eight kernels under it are 8.3 ms of
device time and 133 MB between them — a quarter of the step for 2% of the bytes
— so not one of them is waiting on memory. `rms_norm` is the clearest: 168
dispatches of a `[1, 4096]` row against a `[4096]` weight, 15 µs each to move 35
KB, which is 2 GB/s and a third of a percent of the machine. A decode step gives
that kernel one threadgroup and this GPU has 80 cores. `dense_matmul` is the
same shape of problem at a different width — the routers' `[258, 4096]` gates
are 2.1 MB in 52 µs — and `short_conv`, `router_top_k` and `fused_attention`
join them. **What that says is that the next thing to attack is occupancy and
not bandwidth**, which is the opposite of what a table with only the first row in
it would have said.

**The instrumentation is off by default and the reason is in the numbers.** This
hardware answers `supportsCounterSampling:` with true for `AtStageBoundary` and
false for `AtDispatchBoundary` — Apple silicon offers no timestamp *between* two
dispatches of one compute pass — so a timed dispatch is a compute pass of its
own. What that is deliberately not is a command buffer of its own, which would
put back the round trip two milestones went to remove and measure an engine
nobody runs; the passes still go in the same two submissions. Over seven
alternating pairs it costs **10.4 ms a step and 8.1 ms of device time, 8 µs a
dispatch**, and the pass boundary lands *between* the spans rather than inside
them: the rows above sum to 29.4 ms against the 26.9 ms those same pairs put an
unsampled step's device time at, so each carries a couple of microseconds it
would not have — 9% across the table, and more of it on the short rows than on
the long one. The ranking is the finding; the absolute figures carry that.

**A dispatch's shape is not an allocation.** Each of the 749 dispatches a step
ran at the time took its dimensions, its offsets and the expert its rows go
through in small `MTLBuffer`s of its own — 1374 a step, made and freed between
two steps that wanted the same values — where `setBytes:` puts the same bytes in
the command buffer as the dispatch is encoded.
That took the encode row from 9.35 ms to 7.28 ms, measured against the commit
before it and alternating between the two over seven pairs, with the wait row and
the device's own clock where they were. It stays a *copy* per dispatch rather than
a buffer the layer holds, which is what lets two calls of different heights share
a command buffer — and 953 allocations are left in that row, every one of them an
output or a row copied over for a dispatch that could not read it where it was.

Multiplies that share an input share a command buffer, and so do multiplies that
share nothing: the four projections a layer's normed hidden state feeds, the norm
that makes it, the two convolutions and two head norms behind two of them, the
attention step beside the projection it feeds, the convolution and add on the
residual path behind that, the norm over what they left, and every dispatch the
MLP then runs — 1077 dispatches in two submissions, one for the forty-two layers
and one for the head. **What those have in common is that a seam had to be able to express
them.** Handing a backend one bank at a time, none of it is visible: it takes a
call that is given the whole layer to see that the gate reads the hidden state
the shared bank reads, that the top-k reads what the gate wrote, and that the
routed bank's experts are named by what the top-k wrote. And it takes a backend
holding the *whole* layer to see that the value between `o_proj` and the MLP's
first dispatch is read by nothing else — which is why the two backends this had
are one now, asked four questions about a layer index rather than two about the
model.

Five alternating measurements have now taken 40, 42 or 44 command buffers out of
a step and read the wait row either side, and each says the same thing in a
different denominator: 6.3 ms out of 249, 7.2 out of 209, 6.3 out of 167, 6.6 out
of 127 and 6.7 out of 87 — 157, 172, 156, 165 and 152 microseconds a merged
submission. A submission measured on its own is 225. **The two are different
numbers and this project has to keep them apart**, since only the first says what
merging a command buffer is worth; and no two of the five agree, so an estimate
built on the difference between 152 and 172 is over-fitted.

**A layer's attention became one submission rather than two** for 7.2 ms off the
wait row and 8.8 ms off the step, and its dispatch count went the other way —
581 to 749, four more a layer — for nothing, because what came off beside them
is a copy of the whole cached span onto the device per layer per step. **A MoE
layer became one rather than three** for 12.8 ms off the wait row over two
commits, and its dispatch count rose too, 7 to 10 a layer. Both times the device
was executing for the same time either side, which is what says a merged
submission buys the round trip and not the work.

**A whole layer became one submission rather than two** for 6.7 ms off the wait
row and 8.3 ms off the step, over seven alternating pairs in which every pair
moved the same way: 55.3 ms to 47.0. This is the first of the three where the
device's own clock moved with it — 24.8 ms to 25.9 — because this one did not
only merge command buffers, it moved three operations a layer onto the device to
be able to. That is the trade stated: 1.1 ms more executing, against 6.7 ms less
waiting and 0.8 ms of short convolution that stopped running here at all.

**The attention step is a dispatch even though it has no weight to hand over**,
and what it hands over instead is a tensor nobody builds. The reference adds a
materialised `[B, H, LQ, S]` mask to its logits; the kernel derives each entry
from the backward distance where it scores the key it belongs to, so the mask and
the scores it forces alongside it are never allocated. Over the eight-token
context that profile is taken across, that is a wash — 42 more dispatches cost
about the millisecond the CPU's own scores and mask cost — and it is not what the
kernel is for. A 769-token prompt prefills in 13.6 s against 55.4 s, and the gap
widens with the prompt: ×1.3 at 97 tokens, ×2.4 at 385, ×4.1 at 769.

**A whole decoder layer is now one command buffer**, and twenty-six dispatches
on a layer that routes. Eleven are its attention: the input layernorm, the four
projections that read it, the two short convolutions behind the key and the
value, the two head norms over the query and the convolved key, the attention
step and `o_proj`. Three more are the two residual paths around the MLP — the
layer's two short convolutions, each of which adds the value its block began with
as a second addend where it writes rather than in a dispatch of its own, and the
second norm between them. The last twelve are the MLP: the router's gate, the
top-k over 256 sigmoid-corrected scores, each bank's gate, up, activation and
down, the softmax over the eight logits that selection named, and both banks'
rows weighted by it and summed. Every value between them is a buffer the next
dispatch reads. A dense layer is eighteen, its feed-forward network four where a
MoE layer's two banks and the router around them are twelve.

Neither bank's rows are a tensor anybody builds. A token reads six experts, so
its six rows are one row of the hidden state read six times; every token reads
both shared experts, so the shared bank's rows are the hidden state laid end to
end after itself. `out[i] = x[(i / per_source) % sources] @ w[experts[i]]ᵀ` is
both of those and, at `per_source` of one and `sources` of the rows, every
ordinary dispatch too.

**Five of those twenty-six write state that outlives the call** — four
convolution windows, and the span the step attends over — so where they ran
decided where that state lives, and holding all of it is what let the rest
follow. What a sequence still carries here is a count.

**What that leaves on the CPU is nothing of a layer.** The last two things it
held were the router's own softmax — over eight numbers, where three of the four
ways of misreading this gate live — and the short convolution and residual add
that read what it weighted. The softmax is a dispatch now, measured against
`SparseMoe::weigh`, which stays the arithmetic every fixture holds to mlx-vlm and
is what says the eight weights still come from the raw logits, still span the
routed six and the shared two together, and still carry `route_scale` *and* the
learned `global_scale` whose absence is a 142-fold error. A layer is
`[tokens, hidden]` in and `[tokens, hidden]` out.

**Taking it cost 0.65 ms**, over seven alternating pairs in which every pair
moved the same way: 46.83 ms to 47.48. The 1.8 ms of CPU rows that came off is
real, and so is what replaced it — 122 more dispatches a step is 0.8 ms more
device time and 0.8 ms more encoding, and 162 more buffers made and freed is most
of the rest. The command buffer count did not move, because a layer that is one
command buffer is still one command buffer. What it bought is what a layer is
now: a value in and a value out, and nothing between them anybody names.

**Which is what let the layers themselves merge.** Nothing between layer i's last
dispatch and layer i+1's first forces a wait — what the first writes is what the
second reads and nobody else looks at it — so what forced one was that the seam
above a layer named the value between two of them in this process's memory. It
does not any more: what a layer answers with is either rows or a count of rows
somebody else is holding, and the backend rather than the layer decides which,
because whether what a layer produced crosses back is a question about the layer
*after* it. A run ends where somebody has to read it — a layer the backend does
not hold whole, the end of the stack, or a call of more than one row, which is
where the memory a merged run holds is traded against the round trips it saves.

**So a decode step is two submissions**, one for the forty-two layers and one for
the head, where it was 43 and 87 and 249. Over seven alternating pairs, every
pair moving the same way: 47.43 ms to 34.69. The device's own clock did not move
— 26.7 ms either side — so the 12.7 ms is round trip and nothing else: 10.2 ms of
it off the wait row at 250 microseconds a submission removed, and the rest the 41
uploads and 41 readbacks that stop happening. **250 µs is not the 152 to 172 the
marginal figures had**, and the difference is the serialisation rather than the
submission: a step used to encode a layer, submit it, wait for it, and only then
encode the next. A prefill still submits a layer at a time, and its own numbers
are unchanged.

There is no operation of a layer left outside the GPU. Both backends generate the
same tokens, and the CPU one stays the oracle every kernel here is validated
against.

Or the same model behind an OpenAI-compatible endpoint, loaded once:

    inklingrs serve models/Inkling-Small-mxfp4

    curl -sN http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}'

`POST /v1/chat/completions`, streaming and collected, plus `GET /v1/models`.
Here the turn structure *is* applied — hard-coded rather than interpreted from
`chat_template.jinja`, and checked against what that template renders — because
without it nothing puts the model in a turn it could end and every request runs
to `max_tokens`. The model's thinking channel arrives in `reasoning_content` and
its answer in `content`, with the markers themselves in neither.

One request at a time; a second client waits. Batching is the scheduler's job
and the scheduler is the reason this engine exists.

## Why the reference directory exists

`sconv`, the banded relative-position bias, and sigmoid-gated top-6-of-256
routing cannot be validated by reading generated text. `reference/` is a
patched mlx-vlm used for layer-by-layer tensor comparison — an oracle, not a
dependency of the engine.

Two patches are needed before it loads Inkling-Small at all:

- `03-config-field-names.patch` — mlx-vlm reads `intermediate_size` as the dense
  FFN width, but Inkling calls the dense width `dense_intermediate_size` and uses
  `intermediate_size` for the per-expert width. Unpatched, both are wrong for
  Inkling-Small, and Inkling-975B's two dense layers load at 3072 instead of
  24576.
- `04-drop-identity-expert-scales.patch` — the MXFP4 quant carries identity
  `switch_mlp.{gate,out}_scale` tensors with no counterpart in the model, which
  abort a strict load. Dropped, with a guard that refuses any non-identity value.

The other three — a model-type remap, submodule configs exported for the dump
scripts, and a tap on the MoE router — are what the fixtures are captured
through rather than what makes the checkpoint load.

## Architecture notes

42 layers, hidden 4096, 256 routed experts (top-6) plus 2 shared, 276B total /
12B active. No RoPE — position comes from depthwise causal short convolutions
(kernel 4, on the key and value inside attention and on what attention and the
MLP produced, before each residual add) plus a learned relative logit bias over
a 1024-token extent.

Three properties drive the design:

**Attention is 5:1 local:global.** Layers 5, 11, 17, 23, 29, 35 and 41 are full
attention; the other 35 are capped at a 512-token window. Only the 7 global
layers grow with sequence length, so KV costs 28 KiB/token plus a fixed 70 MiB
per sequence — a 1M-token context fits in under 30 GiB. This is what makes deep
batching plausible on one machine.

**Short-conv state cannot be trimmed.** It keeps only the last `K-1` inputs, so
rejected speculative tokens need restore-and-replay rather than truncation.
Reordering along the batch dimension is fine, so continuous batching works, but
MTP rejection and batching meet here and this is the hard part of the engine.

**The reference materialises the mask.** It builds a full `[B, H, LQ, S]`
additive tensor — acceptable when decoding, quadratic when prefilling, and an
explicit additive mask of that shape also disqualifies MLX's own fused SDPA, so
the scores get spelled out beside it. Together they are 57% of what a
16384-token prefill allocates over the resident weights, and 32768 tokens are
refused at a projected 406 GiB. `--backend metal` builds neither: the
relative-position bias is computed per element inside the attention kernel,
which is where a custom engine wins outright.

## Weights

The MXFP4 quant (`mlx-community/Inkling-Small-mxfp4`, 140 GB) has **no MTP
tensors** — they were stripped during quantisation. It is fine for text, vision,
audio, batching and perf work, but MTP requires the BF16 original
(`thinkingmachines/Inkling-Small`, 532 GB). The official NVFP4 keeps its MTP
weights but is in ModelOpt format, which mlx-vlm cannot read.
