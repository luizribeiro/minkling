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

`just test` is the one to run while iterating: **499 of the 511 tests, no
checkpoint, ten seconds.** Everything a fixture can settle is here — the
kernels against the CPU, the CPU against mlx-vlm's recorded activations, the
tokenizer against the whole vocabulary, the server against its own frames. The
43 that need weights report a skip and pass. It runs through libtest, which puts
a crate's tests in one process: opening a Metal device costs a second, so the 148
kernel tests are 7.5 s sharing a process and 161 s with one each. Nothing in this
tier measures the process it runs in, which is what makes sharing one free.

`just test-full` is what has to pass before a commit lands: **all 511 against a
real checkpoint, four minutes.** The 43 gated tests are what
only weights can settle — that the packed tensors decode to what the reference
decodes, that 42 trained layers reproduce the recorded stack, that the engine
generates the oracle's own continuation, and that it generates the same
continuation while guessing four tokens ahead — and `--backend cpu` is the
oracle they are measured against, at 9.0 s a decoded token, which is where most of those
minutes go. This tier runs a process a test, which is what keeps a test that
bounds its resident set bounding only its own.

`just test-timing` is the twelve tests whose result *is* a number — a duration
they assert on, a resident set they bound, the two decode-step tables quoted
above, what a speculative round costs — run one at a time with nothing beside
them. **A measurement taken while eleven other tests ran is a measurement of
the eleven:** a round trip this repo has at
191 µs reports 598 under a parallel suite, and `.config/nextest.toml` records
what believing a number like that once cost. `#[ignore]` is what keeps them out
of the two runs above, and what selects them here.

Text in, text out, streamed to stdout as each token is decoded:

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' -n 4

A decode step is about 33 ms against mlx-vlm's 32 ms, and the timings go to
stderr so stdout stays pipeable. The prompt reaches the tokenizer as it stands,
so the model *continues* it rather than answering it. A chat turn is written out
in full — `<|message_user|><|content_text|>…<|end_message|><|message_model|>` —
rather than applied by a template this does not implement.

**Every matmul in the model runs on the GPU, and no weight one of them reads is
ever decoded to memory** — the MXFP4 ones in registers a nibble at a time, the
routers' bfloat16 gates by a shift — and `--backend cpu` puts them all back:
0.033 s a token against the CPU's 9.0. The experts were the first two thirds of
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
an eight-token generation at 0.14 GiB and a 769-token prefill at 0.44.

**What a step costs is now mostly the asking, and that is measured rather than
inferred.** Every operation a forward pass runs opens a scope charged the time
inside it that no scope inside *it* claimed, so the rows of a decode step sum to
the step and what they leave over is a number rather than a shrug:

    submit and wait       2    83%      of which the device executed for 24 ms
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
now executing for **90%** of it — two submissions, around work that is no longer
waited for a layer at a time. Nothing an operation of a layer would open a scope
around is left in the table at all: what remains beside the round trip is
encoding it, the sampling at the end, and the embedding at the start.

**And now which kernel owns which of those 24 milliseconds.** The device
timestamps a command buffer, and a decode step is two of them around 1077
dispatches, so until this landed that figure was one number with nine kernels
behind it. It is now nine numbers, each beside the bytes that dispatch said it
moves and what that comes to against this machine's 819 GB/s:

    kernel            calls    device   share       moved   achieved  of peak
    packed_matmul       457   21.07ms   77.8%   5932.36 MB   282 GB/s     34%
    rms_norm            168    1.67ms    6.2%      5.89 MB     4 GB/s      0%
    short_conv          168    1.34ms    5.0%     22.02 MB    16 GB/s      2%
    fused_attention      42     770µs    2.8%      5.62 MB     7 GB/s      1%
    dense_matmul         40     590µs    2.2%     85.24 MB   145 GB/s     18%
    router_top_k         40     544µs    2.0%      0.08 MB     0 GB/s      0%
    swiglu               82     387µs    1.4%      8.26 MB    21 GB/s      3%
    router_weights       40     381µs    1.4%      0.00 MB     0 GB/s      0%
    moe_combine          40     343µs    1.3%      5.90 MB    17 GB/s      2%

**The packed matmul is 78% of the device's time and it is the only kernel here
doing bandwidth's work.** Its 5.9 GB is what the checkpoint's shapes say a token
reads — six of each MoE layer's 256 experts and both shared ones, plus every
layer's own projections — arrived at from a dispatch's own declaration rather
than from that arithmetic, and the two agree. 34% of the machine is not a
finished kernel and it is not a decade off one either; M2's isolated matmul
measured 284 to 424 GB/s, so 282 is the bottom of the range this kernel has
already been seen to reach.

**The other 22% is not waiting on memory, and that was measurable rather than
arguable.** The eight kernels under the matmul are 6.0 ms and 133 MB between
them, and what the first version of this table asked was whether a fifth of a
step for 2% of the bytes meant occupancy. The kernel's own shapes answered:
`rms_norm` normalises a decode step's `[1, 4096]` hidden state as one group and
a query head norm cuts the same 4096 values into 32, and the same values took
17.6 microseconds one way and 5.3 the other, against 1.5 for an empty dispatch
of either grid. **A group is reduced by one threadgroup, which is one core of
eighty.**

**But the remedy was not more threadgroups.** Splitting a group across them
needs a second dispatch to combine what they left, and a dispatch costs 4
microseconds to encode against the 6 the split would save — so what the two
kernels below it actually bought was *more memory in flight per thread*, which
costs no dispatch at all. `rms_norm` reads its group four floats to a lane where
the width allows it and gives each group only the threads its lanes need: 2.53
ms to 1.67, and a `[1, 4096]` norm from 17.6 microseconds to 8.0. `dense_matmul`
reduces each output element across a run of simdgroups rather than one, as long
a run as the dispatch can spare: 2.08 ms to 590 µs, 41 GB/s to 145, and a
decode-shaped gate from 39 microseconds to 10. Neither added a dispatch and
neither moved a token.

**What is left says the same thing in the same voice.** `short_conv` is 5.0% for
2% of the bytes and `fused_attention` 2.8% for 0.7%, and both are already 64 and
32 threadgroups wide at decode — so whatever they are waiting on, it is not the
one core the norm was on, and neither of the two remedies here is theirs. The
matmul is 78% of a step now, which is the first time this table has been mostly
one row.

**The instrumentation is off by default and the reason is in the numbers.** This
hardware answers `supportsCounterSampling:` with true for `AtStageBoundary` and
false for `AtDispatchBoundary` — Apple silicon offers no timestamp *between* two
dispatches of one compute pass — so a timed dispatch is a compute pass of its
own. What that is deliberately not is a command buffer of its own, which would
put back the round trip two milestones went to remove and measure an engine
nobody runs; the passes still go in the same two submissions. Over seven
alternating pairs it costs **11.1 ms a step and 8.9 ms of device time, 8 µs a
dispatch**, and the pass boundary lands *between* the spans rather than inside
them: the rows above sum to 27.1 ms against the 24.4 ms those same pairs put an
unsampled step's device time at, so each carries a couple of microseconds it
would not have — 12% across the table, and more of it on the short rows than on
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
kernel is for. What it is for is the memory the mask would have taken, which the
architecture notes below price.

**What it is not for is prefill wall time, and that belongs to the reference.**
97, 385 and 769 tokens prefill here in 1.87, 5.32 and 10.2 s, best of three,
against 0.42, 0.71 and 1.18 in a single pass of `just prefill-bench` — ×4.5, ×7.5
and ×8.6, and widening with the prompt. Where the gap comes from is unmeasured;
what can be said is that it is not the round trips, since a prefill's submissions
are 42 at 250 µs against a gap of nine seconds. Every milestone here has moved
the decode step, and prefill has never been the path any of them was about.

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
not hold whole, the end of the stack, or a run that has reached the bytes it may
hold, which is where the memory a merged run holds is traded against the round
trips it saves.

**What a run may hold is bytes rather than rows**, because what it holds is every
intermediate of every layer in it until the command buffer completes and a layer
allocates the same buffers whatever the call: a normed state, four projections,
two convolutions, two head norms, what the step and `o_proj` produced, and the
eight expert rows a token routes through. Only their lengths grow with the
tokens. So the backend counts what it has allocated since the run opened —
nothing bound into a command buffer can be freed before it completes, which is
what makes that reading what the run is still carrying — and ends the run when it
reaches a budget. What that budget bounds is exactly what merging adds to a peak:
a run holds the budget plus the layer that crossed it, and that layer holds what
it would have held unmerged. A call whose own layer already reaches the budget
merges nothing and is one submission a layer, which is what a long prefill is.

**The budget was sized by the checkpoint's shapes and is now sized against a
measurement.** A decode step allocates 17.6 MiB across the 953 buffers its one
run retains, so the 160 MiB the budget allows is about nine rows of this stack —
which is the deepest block the eight heads can ask for, and the width the table
under "Speculating with the MTP heads" still submits in two, one for the layers
and one for the head, the same as a single row. It is also why the budget does
not reach a prefill: ten tokens already pass it, so every prompt worth the name
is a submission a layer, exactly as it was.

**So a decode step is two submissions**, one for the forty-two layers and one for
the head, where it was 43 and 87 and 249. Over seven alternating pairs, every
pair moving the same way: 47.43 ms to 34.69. The device's own clock did not move
— 26.7 ms either side — so the 12.7 ms is round trip and nothing else: 10.2 ms of
it off the wait row at 250 microseconds a submission removed, and the rest the 41
uploads and 41 readbacks that stop happening. **250 µs is not the 152 to 172 the
marginal figures had**, and the difference is the serialisation rather than the
submission: a step used to encode a layer, submit it, wait for it, and only then
encode the next. A prefill long enough that one of its layers reaches the budget
still submits a layer at a time, and every prefill worth the name does: 97, 385
and 769 tokens cost 2.04, 5.45 and 10.1 s before the budget replaced the row
count and 1.87, 5.32 and 10.2 s after it, which is the same figure three times
and is the point of where the line was drawn.

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
a rejected speculative token cannot be taken out of it the way a key can:
shortening the window needs the input *before* the ones it holds, and that
input is gone. mlx-vlm's answer is to restore the state and replay the accepted
tokens through the model, which is a whole second forward pass on every round
that rejects one. **Here the window keeps more than it reads** — `slack`
timesteps further back — so a rejection is a shift rather than a replay, and
what it leaves is the window the sequence would have had, bit for bit. See
"Speculating with the MTP heads" below. Reordering along the batch dimension is
fine, so continuous batching works, but MTP rejection and batching meet here and
this is the hard part of the engine.

**The reference materialises the mask.** It builds a full `[B, H, LQ, S]`
additive tensor — acceptable when decoding, quadratic when prefilling, and an
explicit additive mask of that shape also disqualifies MLX's own fused SDPA, so
the scores get spelled out beside it. Together they are 57% of what a
16384-token prefill allocates over the resident weights, and 32768 tokens are
refused at a projected 406 GiB. `--backend metal` builds neither: the
relative-position bias is computed per element inside the attention kernel,
which is where a custom engine wins outright.

## Speculating with the MTP heads

    inklingrs generate models/Inkling-Small-mxfp4 --prompt '…' -n 64 --speculate 2

Inkling ships **eight multi-token prediction heads** and nothing had ever run
them: mlx-vlm drops every `model.mtp.*` tensor at load. A head is a decoder
layer with three tensors in front of it — a norm over the hidden state it is
chained from, a norm over the embedding of the token one position further
ahead, and a `[4096, 8192]` projection that takes the pair back to model width
— and head `d` guesses the token `d + 2` positions on. `reference/results/mtp_acceptance.md`
is the study that settled the wiring; the engine verifies it against the
tensors and reproduces one head against mlx-vlm's own module.

**A round proposes `k`, verifies `k + 1` in one forward pass, and banks the
longest prefix the model agreed with.** Position `i` of the block answers what
follows everything fed up to row `i`, which — while the guesses were right — is
the question the next decode step would have asked, so the first answer that
disagrees is kept too: it is the model's own token from a prefix the model
agrees with. **No guess is ever load-bearing**, and the tokens are identical
with speculation on and off — the recorded continuation of the recorded prompt,
at every depth, asserted in `just test-full` and again over 64 tokens in the
timing tier.

Rejected tokens are taken back rather than replayed: the keys are a counter,
and the four convolution windows a layer keeps are shifted along inside the
slack they were built with. On the device that is the same shift over a buffer
the GPU holds, which unified memory makes a move rather than a copy.

**What a round costs, measured here rather than inherited.** Against a warm
cache, a 34-token prompt, and this engine's own decode step over the 64 tokens
that follow it — 35.6 ms, where the 33 ms above is the same step at the
eight-token context every other measurement in this file is taken at:

    tokens in the block    1      2      3      4      6      9
    forward pass       33.5ms 51.6ms 51.3ms 74.5ms 85.4ms 117.4ms
    × a decode step      0.94   1.45   1.44   2.10   2.40    3.30
    submissions             2      2      2      2      2       2

    heads chained          1      2      3      4      6      8
    the chain           4.8ms  9.7ms 15.8ms 20.3ms 28.4ms  38.3ms
    × a decode step      0.14   0.27   0.44   0.57   0.80    1.08

**An extra token in the block costs 10.5 ms**, which is the acceptance study's
finding reproduced on this engine's own numbers: it measured 10.5 ms against a
31.8 ms step. Most of that is the MoE and is fundamental — one token reads 6
routed experts a layer and nine tokens read up to 54, where the whole bargain of
speculation elsewhere is that verifying `k` tokens costs about what decoding one
does, because you re-read the same weights.

**Every block this engine can propose is two submissions**, where a block of two
or more was 43. A decode step was always two command buffers, because a layer
handed one row can leave what it produced where the next layer reads it; the
engine drew that line at one row, so a call of two paid a submission a layer.
The line is bytes now — nine rows of this stack stay under what a run may retain,
see the layers' own paragraph above — and that is 41 round trips off every block
a round can ask for.

**What those 41 were worth is the finding, and the block table only half
explains it.** At `k = 2` the two agree exactly: the round fell from 83.7 ms to
66.5, and the block of three it verifies fell 1.85 decode steps to 1.44, which is
17.1 ms of the 17.2. At `k = 1` and `k = 3` they do not. Those rounds fell from
63.0 ms to 49.8 and from 115.8 to 86.2, while the blocks they verify barely moved
— 1.46 decode steps to 1.45 at two rows, 2.03 to 2.10 at four. **So a block timed
against a warm cache and the round a generation pays are not the same
measurement**, at two widths out of three; what separates them is unmeasured, and
the sweep is the one that describes a run.

A head's guess costs 4.8 ms, of which about 3.4 is the 950 MB it reads — its own
532 MiB, and `lm_head` again to turn a hidden state into a token — and the rest
is the five submissions a partial handover takes. The study called the
reference's per-head overhead "yours to win in Rust"; most of it turns out to be
bandwidth, and mlx-vlm was already near it at 3.9 ms.

**So every depth pays now.** Over 64 tokens of a structured prompt, three passes
round-robin over the depths so that a drift moves them all, best pass each:

    k                      0      1      2      3      4
    ms/token           35.35  27.22  25.96  28.28  31.50
    tokens a round      1.000  1.829  2.560  3.048  3.368
    speedup             1.000  1.299  1.362  1.250  1.123
    accepted, by depth         85%  91/74% 84/74/63% 82/65/53/47%

**k = 2 is still the depth that pays, at 1.36×**, where it was 1.10× and where
`k = 3` and `k = 4` were losses at 0.95× and 0.86×. Acceptance did not move —
the tokens-a-round row is what it was, to three decimals, because the prompt and
the model are the same — so the whole of it is the round getting cheaper. It
beats the study's projected 1.32× at this depth, which it did not before.

**25.96 ms a token is under mlx-vlm's 32.** The win is speculation's rather than
the kernels': an unspeculated step costs 32.5 ms here against the reference's 32,
so a token this engine decodes on its own is still a shade the slower of the two,
and it is the tokens it does not decode that put the run ahead.

Acceptance is joint rather than marginal and cannot be otherwise in an engine: a
round whose first guess was rejected never learns what its second was worth,
because the position that guess was about is not the position the model went to.
The study's teacher-forced replay could measure both. The spread across
workloads is its headline finding and it holds here — 85% at the first head on
structured text against the 44.9% it measured on prose — so the depth worth
running is the workload's rather than the engine's, and `--speculate` takes it
as a number for that reason.

**The machinery costs a run that does not use it nothing**: 35.57 ms/token
against 35.73 with four timesteps of slack in every window, which is inside the
spread of three passes. A run that speculates nothing maps no head, allocates no
scratch, and asks its windows for no slack at all.

## Weights

The MXFP4 quant (`mlx-community/Inkling-Small-mxfp4`, 140 GB) is what the engine
runs, and the 160 `model.mtp.*` tensors are not in it — they were dropped during
quantisation. **They are not a quantisation of anything.** Every quant that kept
them kept them in bfloat16, and the 8-bit quant's `mtp.safetensors` is the BF16
original's own 160 tensors *byte for byte*, all 4.5 GB of them compared. So the
heads pair with any stack quantised from the same original, and giving this one
its heads is `just mtp-shard` — a file copy, where re-quantising the 532 GB
original to keep them is hours and would write out these same bytes. What it
costs is that the heads see an mxfp4 stack's hidden states rather than the
8-bit stack the acceptance study measured, which is why acceptance is measured
here again rather than inherited.

**So every MTP number in this file was taken on the mxfp4 checkpoint with that
copied shard sitting beside its index, and none of them can be reproduced
without it.** `models/Inkling-Small-mxfp4/mtp.safetensors` is a prerequisite of
the speculation section above rather than a detail of it: absent, the checkpoint
carries no `mtp_config`, `--speculate` has nothing to map, and the timing case
that prices a round refuses to run. The 8-bit quant is not the alternative it
looks like — this engine cannot load it at all, because `quantize.py` leaves
`embed_tokens` and `lm_head` in their original precision and the loader opens
both as packed pairs, so it stops on a missing `embed_tokens.scales`. It is a
checkpoint for the Python acceptance study and for carrying these 4.5 GB, and
that is the whole of what it is for.

No index names that shard, in either quant. The loader maps every
`*.safetensors` in a checkpoint directory and reads the index only for whether a
shard it names is missing. The official NVFP4 keeps its MTP weights but is in
ModelOpt format, which mlx-vlm cannot read.
