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

`just test` is the one to run while iterating: **507 of the 524 tests, no
checkpoint, ten seconds.** Everything a fixture can settle is here — the
kernels against the CPU, the CPU against mlx-vlm's recorded activations, the
tokenizer against the whole vocabulary, the server against its own frames. The
34 that need weights report a skip and pass. It runs through libtest, which puts
a crate's tests in one process: opening a Metal device costs a second, so the 151
kernel tests are 7.8 s sharing a process and 223 s with one each. Nothing in this
tier measures the process it runs in, which is what makes sharing one free.

`just test-full` is what has to pass before a commit lands: **all 524 against a
real checkpoint, eight minutes.** The 43 gated tests — the 34 above and nine of
the measurements below, which need weights as well as a clock — are what
only weights can settle — that the packed tensors decode to what the reference
decodes, that 42 trained layers reproduce the recorded stack, that the engine
generates the oracle's own continuation, and that it generates the same
continuation while guessing four tokens ahead — and `--backend cpu` is the
oracle they are measured against, at 9.0 s a decoded token, which is where most of those
minutes go. This tier runs a process a test, which is what keeps a test that
bounds its resident set bounding only its own.

`just test-timing` is the seventeen tests whose result *is* a number — a duration
they assert on, a resident set they bound, the three decode-step tables quoted
above, what a speculative round costs — run one at a time with nothing beside
them. **A measurement taken while fifteen other tests ran is a measurement of
the fifteen:** a round trip this repo has at
191 µs reports 598 under a parallel suite, and `.config/nextest.toml` records
what believing a number like that once cost. `#[ignore]` is what keeps them out
of the two runs above, and what selects them here.

Text in, text out, streamed to stdout as each token is decoded:

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' -n 4

A decode step is about 20 ms against mlx-vlm's 23 ms, and the timings go to
stderr so stdout stays pipeable. The prompt reaches the tokenizer as it stands,
so the model *continues* it rather than answering it. A chat turn is written out
in full — `<|message_user|><|content_text|>…<|end_message|><|message_model|>` —
rather than applied by a template this does not implement.

**Every matmul in the model runs on the GPU, and no weight one of them reads is
ever decoded to memory** — the MXFP4 ones in registers a nibble at a time, the
routers' bfloat16 gates by a shift — and `--backend cpu` puts them all back:
0.029 s a token against the CPU's 9.0. The experts were the first two thirds of
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

    submit and wait      30    66%      of which the device executed for 18 ms
    dispatch encode    1077    27%
    readback              2     0%
    everything else                     7%

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
paragraph below. Two thirds of a step is a wait and the device is executing for
**90%** of the step, which is a share above the row it sits in: a run of layers
commits part way through and keeps encoding, so a command buffer executes while
this process is charging its time to `dispatch encode`. Nothing an operation of a
layer would open a scope around is left in the table at all: what remains beside
the round trip is encoding it, the sampling at the end, and the embedding at the
start.

**"Three quarters of a step is a round trip" was a sentence worth reading twice,
because a milestone that read it as three quarters of a step spent asking would
have gone and removed submissions.** The driver timestamps four points on every
command buffer and the wait can be divided by them, one row per shape of
submission rather than summed. What it said when a step was two of them:

    dispatches   a step     waited  scheduled    queued   executed  unattributed
    1                 1     1.98ms    54.49µs   65.54µs   656.22µs        1.21ms
    1076              1    18.75ms   934.09µs  106.36µs    17.53ms      183.68µs

`scheduled` is the driver turning a committed buffer into work the GPU can start
and `unattributed` is what none of the three claim — the commit reaching the
driver, and this thread being woken once the buffer completed. **The big
submission was 93% execution and there was nothing in it to remove.** The 1.2 ms
that was not execution is mostly the driver walking 1076 dispatches at 0.87 µs
each, which is a cost of *having* the dispatches rather than of submitting them.
The head's submission is the other kind, and still is: 1.98 ms of wait around
0.66 ms of work, so 1.3 ms of it buys nothing, and that is the price of the seam
that reads the stack's rows back to norm them on this side.

**What was not in that table is the encode, and that was the finding.** A command
buffer executes nothing until it is committed, and a step committed after
encoding all 1076 of the stack's dispatches — so its 4.4 ms `dispatch encode` row
was 4.4 ms with the GPU idle, ahead of the wait rather than inside it. **So the
run commits at the first layer boundary past 64 dispatches now and carries on
encoding into the next buffer**, waiting for none of them until somebody reads
the rows. A MoE layer is 26 dispatches, so that boundary is three of them, and
the same table reads:

    dispatches   a step     waited  scheduled    queued   executed  unattributed
    1                 1   983.97µs    52.68µs  146.65µs   659.26µs      125.38µs
    52                1   842.85µs    68.56µs   11.14ms   839.12µs        0.00ns
    78               12    11.27ms     1.13ms   70.83ms    15.05ms        0.00ns
    88                1     1.07µs   106.93µs  424.15µs     1.55ms        0.00ns

**A `queued` column of 71 ms inside a 20 ms step is the whole of what changed.**
Twelve command buffers sit in the queue while the ones ahead of them run, and the
one this process blocked for 1.07 microseconds is the first of them — committed
and finished before there was anything else to wait for. `unattributed` is
nothing on every row a run committed, because the three parts of those now
account for more than the wait rather than less.

Over seven alternating pairs, every pair moving the same way and the two ranges
not overlapping, a decode step is **26.34 ms to 20.11**. **The device's own clock
did not move** — 18.13 ms against 18.10 — so all 6.2 ms of it is this process's
wait and none is work the GPU stopped doing, which is what says the change is
scheduling and nothing else. A second seven pairs taken before the device's clock
was read beside the step put the same figures at 26.39 and 20.18. The recorded
continuation did not change, and neither did the peak resident set.

**And now which kernel owns which of those 18 milliseconds.** The device
timestamps a command buffer, and a decode step is fifteen of them around 1077
dispatches, so until this landed that figure was one number with nine kernels
behind it. It is now nine numbers, each beside the bytes that dispatch said it
moves and what that comes to against this machine's 819 GB/s:

    kernel            calls    device   share       moved   achieved  of peak
    packed_matmul       457   15.20ms   70.3%   5932.36 MB   390 GB/s     48%
    rms_norm            168    1.88ms    8.7%      5.89 MB     3 GB/s      0%
    short_conv          168    1.37ms    6.3%     22.02 MB    16 GB/s      2%
    fused_attention      42    880µs     4.1%      5.62 MB     6 GB/s      1%
    dense_matmul         40    594µs     2.7%     85.24 MB   143 GB/s     18%
    router_top_k         40    547µs     2.5%      0.08 MB     0 GB/s      0%
    swiglu               82    437µs     2.0%      8.26 MB    19 GB/s      2%
    router_weights       40    378µs     1.7%      0.00 MB     0 GB/s      0%
    moe_combine          40    351µs     1.6%      5.90 MB    17 GB/s      2%

**The packed matmul is 70% of the device's time and it is the only kernel here
doing bandwidth's work.** Its 5.9 GB is what the checkpoint's shapes say a token
reads — six of each MoE layer's 256 experts and both shared ones, plus every
layer's own projections — arrived at from a dispatch's own declaration rather
than from that arithmetic, and the two agree. 48% of the machine is what a lane
holding four packed bytes rather than one is worth, against 34% before it — and
390 GB/s is three quarters of the way up the 284-to-424 GB/s M2's isolated matmul
measured, where 282 sat at the bottom of it. Its own paragraph below says what
is left.

**The other 30% is not waiting on memory, and that was measurable rather than
arguable.** The eight kernels under the matmul are 6.4 ms and 133 MB between
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

**What is left was diagnosed rather than attacked, and the two rows turned out
not to be the same kind of row at all.** Both are already 64 and 32 threadgroups
wide at decode, so K1 was right that neither waits on the one core the norm was
on. What separates them is what an *empty* dispatch of the same grid costs — a
kernel that returns on its first instruction, which is everything except the
work:

    short_conv, decode      grid    a dispatch   launch    the convolution
    key or value        16 groups        3.63µs   1.42µs                61%
    residual path       64 groups        4.51µs   2.01µs                55%

    fused_attention     32 groups    a dispatch   launch        the step
    over     8 keys                       9.31µs   1.81µs             80%
    over    97 keys                      74.60µs   1.44µs             98%
    over   512 keys                     368.50µs   1.90µs             99%
    over  4096 keys                       2.57ms   1.29µs            100%

**`short_conv` is a dead end and the numbers say how much of one.** Four a layer
at those two shapes is 684 µs a step measured on their own, against the 1.37 ms
the table charges them — which is 6.3% of the device's *sampled* time, 1.14 ms
once the sampling bias comes off, and 5.7% of a 20.11 ms step. Why a dispatch
measured beside its own kind costs less than the same dispatch measured inside a
step is not attributed here. Either way about two fifths of it is a launch that a
dispatch of any grid pays, so what a perfect convolution could reach is 2.0% to
3.4% of a decode step — a larger share than it was, because the step it is a
share of shrank by a quarter and the kernel did not. The only thing that removes a launch is removing a
dispatch, and the four are four because they convolve four different things.
**Nothing here was changed.**

**`fused_attention`'s 4.1% is not a property of the kernel.** The launch is under
2% of it past a hundred keys and the rest tracks the span: the marginal key costs
0.62 µs between 97 and 4096, and the per-key figure falls from 1.18 µs at eight
keys to 0.63 at four thousand as a 32-key tile stops being mostly empty. So what
that row reports is the *context the profile is taken at*, which is the recorded
prompt and eight generated tokens. Read off the rows above, the same 42 dispatches
are 4.9 ms at the 155-key context the paired measurement below decodes over, and
15.5 ms at 512. The kernel is not waiting on occupancy and it is not waiting on a
floor. It is walking the span — and it was walking all of it, on 35 layers that
cap at a 512-token window. That is M9's deferred finding arriving as this one's
diagnosis, and where the work went: neither of the two remedies this table has
used before, but the loop bound whose table is under the attention step above.
**It buys nothing at the context this profile is taken at and nothing at the one
the comparison below decodes over** — the window is 512 and neither reaches it —
which is why this row is where it was and why the change is a prefill and
long-context one that a decode step pays a compare for.

The matmul is 70% of the device's time, which is what keeps this table mostly one
row.

**And the row that is most of the table wanted the same remedy a third time.**
A lane of the packed matmul read one byte of its weight row, and around that
byte it also read the byte's group scale and the two inputs the byte's two codes
multiply — four memory instructions, of which only one asked for a byte the
dispatch is charged. A lane holding four bytes issues those same four for eight
codes, because a scale covers sixteen bytes and the eight inputs are consecutive
floats. Over seven alternating pairs in which every pair moved the same way, a
decode step went 29.65 ms to 23.85 and the device's own clock went with it,
23.84 to 18.18; the kernel's row went 21.58 ms to 15.66, 275 GB/s to 375. No
token moved.

**What the table says next is that the shapes disagree, and that acting on it
buys nothing yet.** Over one synthetic bank small enough to stay in cache — so
the figures rank the shapes rather than state a bandwidth — this kernel reaches
570 GB/s at a routed bank's decode shape and 700 at a prefill's, against 170 at a
`[1, 4096] @ [1024, 4096]ᵀ` key projection and 380 at the two `[4096, 4096]`
ones: the shapes with too few output elements to fill eighty cores. What a cold
weight costs instead is the 299 GB/s one dispatch of `lm_head` measures, and the
375 the whole step does. Both of the
levers that answered that elsewhere were measured here and neither pays.
`dense_matmul`'s run of simdgroups over one output element is worth 18% on those
same small shapes and gives up 29% on a routed bank's down projection and 11% on
a prefill, which is the wrong side of every byte that matters. Eight bytes to a
lane is worth 14% on the key projection and gives up 4% on the routed banks that
are 55% of a step's bytes, and the step says the two cancel exactly: 18.14 ms of
device time against 18.12 over five alternating pairs, which is no number at all.
So the width is one number rather than a rule, and the next thing to try is not a
wider lane.

**The instrumentation is off by default and the reason is in the numbers.** This
hardware answers `supportsCounterSampling:` with true for `AtStageBoundary` and
false for `AtDispatchBoundary` — Apple silicon offers no timestamp *between* two
dispatches of one compute pass — so a timed dispatch is a compute pass of its
own. What that is deliberately not is a command buffer of its own, which would
put back the round trip two milestones went to remove and measure an engine
nobody runs; the passes still go in the same command buffers. Over seven
alternating pairs it costs **12.1 ms a step and 10.0 ms of device time, 9 µs a
dispatch**, and the pass boundary lands *between* the spans rather than inside
them: the rows above sum to 22.0 ms against the 18.2 ms those same pairs put an
unsampled step's device time at, so each carries a couple of microseconds it
would not have — 21% across the table, and more of it on the short rows than on
the long one. **The bias grew in both denominators and why is unmeasured**: 3.8
ms of over-reporting where it was 2.7, over the same 1077 boundaries, so a
boundary costing what it costs whatever is inside it does not explain it. The
ranking is the finding; the absolute figures carry that.

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
MLP then runs — 1077 dispatches, and the only thing that ends a command buffer
is a run reaching the dispatches it commits at or the head reading the rows. **What those have in common is that a seam had to be able to express
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

**And the mask it derives is now a loop bound rather than an entry it computes
and throws away.** The kernel scored every key of the span and let the softmax
discard the ones the band ruled out. Two of the band's four branches are decided
by the distance alone — nothing at or after the query is causal, and on a
windowed layer nothing further back than the window is in it — so both are the
same comparison made once for the row instead of once for every key of it. 35 of
the 42 layers have a 512-token window and those keys are most of a long span:

    one query, 32 threadgroups          bounded   walking the span whole
      512 keys, global                 368.30µs                 368.67µs
     4096 keys, global                   2.57ms                   2.58ms
      512 keys, window 512             369.05µs                 369.61µs
     4096 keys, window 512             388.67µs                   2.51ms
    16384 keys, window 512             398.50µs                   9.74ms

**Exactly nothing where nothing is outside the window, and ×24 where the span is
32 times it** — which is the shape of the finding rather than a rate, since what
a windowed layer now pays is flat in the context and what it paid before was
linear. The two kernels are kept side by side rather than one of them being a
number in a commit message, and the same pairing is what says the answer did not
move: **bit for bit** over ten cases and 1.95 million query-key pairs the band
masks, including the 1280-query capture. That is exact rather than within a
tolerance because the bound starts on a tile boundary — a tile's softmax rescales
by a maximum over what the tile holds, so tiles cut in different places land a
few ulps apart, and one extra tile of keys already inside the window is what buys
the stronger claim.

**What it is not for is prefill wall time, and that belongs to the reference.**
97, 385 and 769 tokens prefill here in 1.55, 4.68 and 8.37 s against the
reference's 0.256, 0.680 and 1.132 — ×6.1, ×6.9 and ×7.4, and widening with the
prompt. **Three things have moved this row and all three were about something
else**: the matmul took it 1.90, 5.39 and 10.14 s to 1.75, 4.70 and 8.87; the
loop bound was written for a long context and paid at a short one, taking it to
1.73, 4.69 and 8.33; and pipelining a decode step's run took the shortest length
to 1.55 over three alternating pairs, all three moving the same way, while the
other two lengths did not move — 4.83 s to 4.68 and 8.39 to 8.37. That last pair
re-measured both sides rather than reading the middle stage's figures off this
file, which is why its `before` is 4.83 and 8.39 where the row above records 4.69
and 8.33: a sitting a milestone apart is a different sitting. That is where the
budget is drawn: a 97-token prefill still merges four layers to a run and a
longer one merges none, so only the shortest has a run to pipeline. The peak
resident set at the longest is 0.434 GiB.

**And now where those seconds go, which is a question this file has answered for
a decode step and had never once asked of the other regime.** The same
sampling, the same table, at the two longer lengths above — one sitting, in
which those two prefilled unsampled in 4.62 and 8.37 s:

    kernel                    385 tokens                769 tokens
                        device   share    moved     device   share     moved
    packed_matmul        3.32s   94.3%  2116 GB      6.72s   91.3%   4227 GB
    fused_attention    153.6ms    4.4%  0.70 GB    549.5ms    7.5%   1.39 GB
    dense_matmul        20.1ms    0.6%  0.35 GB     38.7ms    0.5%   0.62 GB
    swiglu               9.4ms    0.3%  3.18 GB     18.5ms    0.3%   6.35 GB
    short_conv          10.2ms    0.3%  1.87 GB     17.8ms    0.2%   3.72 GB
    rms_norm             4.6ms    0.1%  1.72 GB      7.7ms    0.1%   3.44 GB
    moe_combine          3.1ms    0.1%  2.27 GB      6.4ms    0.1%   4.54 GB
    router_top_k        0.85ms    0.0%  0.02 GB      1.4ms    0.0%   0.03 GB
    router_weights      0.55ms    0.0%  0.00 GB     0.48ms    0.0%   0.00 GB

**The matmul is 94% of a prefill's device time where it is 70% of a decode
step's, and it is not slow — it is reading the model once per token.** 2116 GB at
385 tokens and 4227 at 769 is 5496 MB a token at both lengths, and the figure a
*decode* step moves with its head taken off is 5495 MB. So the two regimes read
exactly the same bytes per token: a 769-token prefill reads all 42 layers' active
weights 769 times over, which is what 769 decode steps would have read, and the
one thing a prefill is for — reading a weight once and multiplying it against
every row that wants it — does not happen here at all. The linearity is exact
rather than approximate: the totals are in the ratio 1.9972 where the token
counts are 1.9974.

**The bandwidth column says the same thing from the other side.** This kernel
reaches 638 and 629 GB/s at the two prefill lengths — 78% and 77% of this
machine's 819 — against the 386 GB/s and 47% it reaches at decode. **It is
nearer the machine at prefill shape than anywhere else this file measures**, so
there is nothing in the kernel's own execution to win back. The whole of the gap
is byte count, and the byte count is the model's own arithmetic: a row of this
dispatch is a whole weight — see `PackedBank::moves` — and a prefill hands it one
row per token rather than one row per weight.

**Which is what says where the bytes are, and it is not spread evenly.** By the
checkpoint's shapes, a token's 5.5 GB is 59.1% the routed banks it reads six of,
19.7% the two shared banks every token reads, 17.2% every layer's own five
projections and 3.9% the two dense layers' feed-forward network. Rows that could
share a weight read are the ones naming the same expert, and only the last three
of those four have any: a projection's rows all name expert zero and a shared
bank's name one of two, where the routed bank's six rows a token are six
different experts by construction. **So 40.8% of a prefill's bytes are reachable
without moving a row and 59.1% are not**, and that is the line the commit after
this one is drawn on.

**M9's hypothesis did not hold, and the table is what says so.** It was that
every `(head, query)` threadgroup re-reading all keys is the next order of
magnitude here. `fused_attention` is 4.4% of a 385-token prefill's device time
and 7.5% of a 769-token one — growing with the prompt, as the hypothesis
implies, and two orders below where the time is. A perfect attention kernel that
cost nothing at all would take 8.37 s to 7.82. It stays worth doing eventually
and it is not what a prefill is waiting on.

**Nor is it the round trips, and this time that is measured rather than argued
from a submission count.** A prefill is one submission a layer — 42 of them at
385 tokens and 43 at 769, since one of its layers alone passes the bytes a merged
run may hold — and the wait on each is 97% and 98% execution, with `queued` at
nothing on every layer's row. So M16's pipelining reaches none of this: there is
no second command buffer behind the one being waited for, because a prefill never
merges two layers into one. The 42 submissions are 250 µs against a gap of seven
seconds, which is where that argument stood before, and the round-trip table now
says it rather than the arithmetic.

**One thing in the diagnosis is unexplained and is written down as such.** A
sampled prefill is *faster* in wall time than an unsampled one — 3.57 s against
4.62 and 4.65 at 385 tokens, 7.26 against 8.37 and 8.29 at 769 — while the
device's own clock does not move: 3.36 s against 3.32 and 3.36, and 6.96 s
against 6.96 and 6.87. It is not the first read of a length faulting its pages
in, because the two unsampled runs sit either side of the sampled one and agree
with each other and with the figure the command line reports cold. So about a
second of a prefill is this process's own, it is not execution, and putting each
dispatch in a compute pass of its own removes it. Nothing here has attributed
that, and no
number above rests on it — every row in the tables is device time.

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
under "Speculating with the MTP heads" still submits in fifteen, fourteen for the
layers and one for the head, the same as a single row. It is also why the budget does
not reach a prefill: ten tokens already pass it, so every prompt worth the name
is a submission a layer, exactly as it was.

**So a decode step became two submissions**, one for the forty-two layers and one
for the head, where it was 43 and 87 and 249 — and is fifteen now for a reason
that has nothing to do with round trips, since it still waits once. Over seven alternating pairs, every
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
and is the point of where the line was drawn. Those two are that commit's own
pair rather than what a prefill costs today — the prefill section above is.

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
that follow it — 21.3 ms, where the 20.1 ms above is the same step at the
eight-token context every other measurement in this file is taken at:

    tokens in the block    1      2      3      4      6      9
    forward pass       26.3ms 30.5ms 38.4ms 45.8ms 62.2ms  87.7ms
    × a decode step      1.23   1.43   1.80   2.15   2.92    4.12
    submissions            15     15     15     15     15      15

    heads chained          1      2      3      4      6      8
    the chain           4.9ms  9.6ms 14.4ms 19.2ms 28.7ms  37.6ms
    × a decode step      0.23   0.45   0.68   0.90   1.35    1.77

**An extra token in the block costs 7.7 ms**, which is 0.36 of a decode step
against the 0.33 the acceptance study measured — 10.5 ms against a 31.8 ms step.
Both the block and the step it is weighed against fell; the fraction rose,
because the step fell further. Most of that is
the MoE and is fundamental — one token reads 6 routed experts a layer and nine
tokens read up to 54, where the whole bargain of
speculation elsewhere is that verifying `k` tokens costs about what decoding one
does, because you re-read the same weights.

**Every block this engine can propose is one round trip**, where a block of two
or more was 43. A decode step was always two command buffers, because a layer
handed one row can leave what it produced where the next layer reads it; the
engine drew that line at one row, so a call of two paid a submission a layer.
The line is bytes now — nine rows of this stack stay under what a run may retain,
see the layers' own paragraph above — and that is 41 round trips off every block
a round can ask for. The fifteen in the row are command buffers a run commits as
it fills them and waits for once at the end, which is a different number from the
one that used to be the same number.

**What those 41 were worth is the finding, and the block table only half
explains it.** At `k = 2` the two agree exactly: the round fell from 83.7 ms to
66.5, and the block of three it verifies fell 1.85 decode steps to 1.44, which is
17.1 ms of the 17.2. At `k = 1` and `k = 3` they do not. Those rounds fell from
63.0 ms to 49.8 and from 115.8 to 86.2, while the blocks they verify barely moved
— 1.46 decode steps to 1.45 at two rows, 2.03 to 2.10 at four. **So a block timed
against a warm cache and the round a generation pays are not the same
measurement**, at two widths out of three; what separates them is unmeasured, and
the sweep is the one that describes a run. Those six figures are that commit's
own pair; the tables above are what the two cost now.

A head's guess costs 4.7 ms, of which about 3.4 is the 950 MB it reads — its own
532 MiB, and `lm_head` again to turn a hidden state into a token — and the rest
is the five submissions a partial handover takes. The study called the
reference's per-head overhead "yours to win in Rust"; most of it turns out to be
bandwidth, and mlx-vlm was already near it at 3.9 ms — on the 8-bit checkpoint,
whose heads are these heads byte for byte but whose `lm_head`, which a guess also
reads, its quantiser left in the original precision. The two figures are close
for reasons only half of which are shared.
**Only the `lm_head` half
of those bytes is the packed matmul's**, and reading four to a lane took that
dispatch 1.57 ms to 1.46 and the guess 4.85 to 4.70 — the same tenth of a
millisecond arriving twice, which is what says the head's own bfloat16 tensors
are the other half and no kernel here has been at them.

**Speculation has stopped paying.** Over 64 tokens of a structured prompt, three
passes round-robin over the depths so that a drift moves them all, best pass
each:

    k                      0      1      2      3      4
    ms/token           21.33  20.06  21.27  24.04  28.06
    tokens a round      1.000  1.829  2.560  3.048  3.368
    speedup             1.000  1.063  1.003  0.887  0.760
    accepted, by depth         85%  91/74% 84/74/63% 82/65/53/47%

**Every absolute figure here improved but one, and the whole speedup column
collapsed anyway**, which is what a fixed cost looks like when the thing it was
hiding gets cheaper. The same sweep re-run against the commit before this one —
both sides in one sitting, rather than against the figures this file already had
— read 30.43, 23.77, 23.16, 24.83 and 27.99 — so the unspeculated step fell by 30% and `k = 2`
by 8%, while `k = 4` went the other way by 0.07 ms and is the one row here that
did not gain,
because a round's fixed costs are the eight heads' chain and the block's own
extra rows and the chain did not move at all: 37.92 ms to 37.63 over eight heads.
A head is five submissions of a partial handover — see the heads' own paragraph
above — and none of the five is a run of layers, so nothing this milestone did
reached them. Acceptance is untouched, to three decimals, because the prompt and
the model are the same.

**`k = 1` is what pays now and it pays 1.06×**, where `k = 2` paid 1.31× in the
sweep this one is measured against.
The engine is faster at every depth a run can be given and there is now barely a
depth worth giving it, which is the honest reading: the next thing speculation
needs is the heads' chain, not the verify block.

**Against mlx-vlm, measured in one sitting on 2 August 2026.** Both engines were
given the same 27-token prompt — the string mlx-vlm's own chat template renders
for `just smoke`'s question — and both decoded 128 tokens from it. The two
continuations are the same 128 tokens, byte for byte, which is what makes this a
comparison of two engines rather than of two workloads. Six rounds, the order of
the two halves flipped each round so that neither always ran on the other's warm
page cache:

    round                 1      2      3      4      5      6     mean
    mlx-vlm ms/token  22.62  22.73  22.62  22.62  22.62  22.73    22.66
    ours, k = 0       22.06  22.07  22.08  22.07  22.10  22.06    22.07
    ours, k = 2       26.73  27.02  27.07  26.85  26.96  26.61    26.87

**So this engine decodes at 0.97× the reference unspeculated, and it is ahead by
about 3%.** The same sitting before the run was pipelined read 29.29 against
22.77 — 1.29× and behind — and what closed a gap of six and a half milliseconds
is that the device now runs a step's first layers while this process is still
encoding its last. Every one of the six rounds falls the same way and the two
ranges do not overlap, which at a 3% margin is the whole of what makes it a
claim rather than a coin.

**And `k = 2` is now a loss on this prompt**, at 26.87 against 22.07. It was
never a win against the reference — 26.72 against 22.77 was 1.17× behind — but it
was worth 1.10× against this engine's own unspeculated step, and it is worth
0.82× against it now. Nothing about the speculation changed — the same
128 tokens come out at `k = 1`, `2` and `4` as at `k = 0`, byte for byte — and
this prompt's first head is accepted 66% of the time against the sweep prompt's
85%, which was never enough to pay for a chain of heads at the step this engine
has now.

**Both engines drifted over the sitting, and neither much.** The reference held
at 22.62 to 22.73 ms and this engine at 22.06 to 22.10, where the sitting before
this one had the reference 1.1% slower across its six rounds and this engine 1.7%
faster. Two figures taken an hour apart would have carried the sum of those in
whichever direction the order chose, which is the whole argument for alternating
rather than measuring one engine and then the other. Swap was at zero and free
memory at 138 GiB when the sitting opened, the GPU was idle before the first
round, and the four vllm-mlx daemons `reference/results/prefill.md` already counts
were resident at 41 GiB between them.
The sitting before this one recorded two things about the reference that this one
had no reason to disturb: two of its twelve runs prefilled their own 27-token
prompt at 27.8 tok/s against 196–202 for the other ten, a 7× swing inside the
process that never reached its decode rate; and its model load was 6.5–7.1 s
while its pages were in the buffer cache and 20.7 s once the 8-bit checkpoint had
evicted them.

**The reference never moved, and what looked like it moving was the checkpoint.**
`reference/results/mtp_acceptance.md` records a 31.8 ms reference decode step, and
that study ran on `Inkling-Small-8bit`. Measured back to back in this sitting, the
same script on the same host reads 44.0 tok/s — 22.7 ms — against the mxfp4
stack's 140 GB and 33.1 and 33.2 over two runs — 30.2 ms — against the 8-bit
stack's 282 GB, reproducing the study's figure to 5%. So the 33% this file
previously reported as an unexplained change in the reference is **2.01× the
weight bytes buying 1.33× the step**. Both engines here are read against mxfp4 and
always were; nothing on this side touches the reference either way.

Acceptance is joint rather than marginal and cannot be otherwise in an engine: a
round whose first guess was rejected never learns what its second was worth,
because the position that guess was about is not the position the model went to.
The study's teacher-forced replay could measure both. The spread across
workloads is its headline finding and it holds here — 85% at the first head on
structured text against the 44.9% it measured on prose — so the depth worth
running is the workload's rather than the engine's, and `--speculate` takes it
as a number for that reason.

**The machinery costs a run that does not use it nothing**: 28.98 ms/token
against 28.82 with four timesteps of slack in every window, which is inside the
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
