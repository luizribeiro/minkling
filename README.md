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

`just test` is the one to run while iterating: **523 of the 547 tests, no
checkpoint, ten seconds.** Everything a fixture can settle is here — the
kernels against the CPU, the CPU against mlx-vlm's recorded activations, the
tokenizer against the whole vocabulary, the server against its own frames. The
34 that need weights report a skip and pass. It runs through libtest, which puts
a crate's tests in one process: opening a Metal device costs a second, so the 163
kernel tests are 7.9 s sharing a process and minutes with one each. Nothing in this
tier measures the process it runs in, which is what makes sharing one free.

`just test-full` is what has to pass before a commit lands: **all 547 against a
real checkpoint, ten minutes.** The 45 gated tests — the 34 above and eleven
of the measurements below, which need weights as well as a clock — are what
only weights can settle — that the packed tensors decode to what the reference
decodes, that 42 trained layers reproduce the recorded stack, that the engine
generates the oracle's own continuation, and that it generates the same
continuation while guessing four tokens ahead — and `--backend cpu` is the
oracle they are measured against, at 9.0 s a decoded token, which is where most of those
minutes go. This tier runs a process a test, which is what keeps a test that
bounds its resident set bounding only its own.

`just test-timing` is the twenty-four tests whose result *is* a number — a duration
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
    router_top_k         40    547µs     2.5%      0.08 MB     0 GB/s      0%
    dense_matmul         40    480µs     2.2%     85.24 MB   178 GB/s     22%
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

**And that same kernel then took the lever a second time, for a reason that had
nothing to do with a decode step.** A lane read one bfloat16 value, which is two
byte loads and the input float it multiplies — three memory instructions
carrying one multiply-add. Four values to a lane makes that twelve carrying
four, and its row here is 480 µs at 178 GB/s where it was 594 at 143. That is
114 microseconds of an 18 ms device step and is not why it was done: this kernel
is 0.06% of the checkpoint here and 68% of a chain of MTP heads, whose weights
the quantiser left alone. See "Speculating with the MTP heads" below.

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
97, 385 and 769 tokens prefill here in **1.22, 3.20 and 5.52 s against the
reference's 0.26, 0.68 and 1.14** — ×4.7, ×4.7 and ×4.8, both sides measured in
the one sitting, in two rounds with the order of the two halves flipped. **The
gap has stopped widening with the prompt**, which is the first sitting in this
file of which that is true: the same two rounds put the commit before this one
at ×5.0, ×5.6 and ×5.9, so what the column tile below took off is the slope
rather than a constant. The reference did not move between the rounds — 0.26,
0.68 and 1.14 both times, to the hundredth.

**Six things have moved this row and only the last three were about it**: the
matmul took it 1.90, 5.39 and 10.14 s to 1.75, 4.70 and 8.87; the loop bound was
written for a long context and paid at a short one, taking it to 1.73, 4.69 and
8.33; pipelining a decode step's run took the shortest length to 1.55 while the
other two did not move, because a 97-token prefill still merges four layers to a
run and a longer one merges none; then the row tile, 1.53 s to 1.45, 4.75 to
4.17 and 8.40 to 7.78; then the grouping, 1.47 s to 1.33, 4.06 to 3.84 and 7.66
to 6.62; and then the column tile below — over four alternating pairs a length in
which every pair moved the same way and no two ranges overlapped, **1.31 s to
1.22, 3.78 to 3.20 and 6.60 to 5.52**, ranges 1.27-1.33 against 1.20-1.23,
3.67-3.86 against 3.14-3.25 and 6.53-6.69 against 5.44-5.60, and the device's own
clock 0.59 to 0.50, 2.40 to 1.83 and 5.11 to 3.82. Each pair re-measures both
sides rather than reading the stage before it off this file, which is why the
last one's `before` is 1.31, 3.78 and 6.60 where the row above records 1.33, 3.84
and 6.62: a sitting a milestone apart is a different sitting.

**What a prefill holds is what it allocates, and the grouping is the only change
to it that allocated anything at all.** A tile is registers, on both of its
axes; a permutation is two `uint` a row that the sort writes and the bank reads.
That is 1.4 MiB of the 13741.2 MiB a 769-token prefill allocates, 80 buffers of
the 1278, and it is held for as long as the layer's command buffer is — where
the layer's own intermediates at that length are 37.8 MB apiece. The column tile
adds nothing to either column: 1686.5, 6873.3 and 13741.2 MiB over 1127, 1157
and 1278 buffers, the same figures to the tenth of a mebibyte either side of it,
because sixteen running sums a lane are sixteen registers and not a buffer. The
bound a merged run is traded against does not reach a prefill and still does
not: ten tokens already pass it, so every prompt worth the name is a submission a
layer, and 42 and 43 submissions are what they were.

**And the decode step did not move, which is the constraint all of this was done
under.** Over seven alternating pairs with the order alternating too, a step is
20.21 ms against the 20.49 the commit before the column tile reads, and the
device's own clock 18.17 against 18.40 — with the two ranges lying across each
other, 20.00-20.96 against 20.02-21.30, and three of the seven pairs falling the
other way. By this file's own standard for a real effect — every pair moving the
same way and the ranges not overlapping — that is no number at all, and the
grouping before it read the same way. What says *why* is not the timing: `tiles`
and `groups` are both false for every shape a decode step dispatches, so the
table of a decode step's kernels has the same 457 `packed_matmul` calls moving
the same 5932.36 MB and **neither a `packed_matmul_rows` row nor a
`packed_matmul_grouped` one**, in the same 1077 dispatches, 15 submissions and
953 buffers of 17.6 MiB. The recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` did not change; nor did 48
tokens of a longer prompt, byte for byte across the two builds at every depth
speculation runs at — `k` of 0, 1, 2 and 4 — nor what `--backend cpu` answers,
which no kernel change can reach and which is checked rather than assumed.

**And now where those seconds go, which is a question this file has answered for
a decode step and had never once asked of the other regime.** The same sampling,
the same table, at the two longer lengths above, and **as it stood before the
tile further down** — one sitting, in which those two prefilled unsampled in
4.62 and 8.37 s:

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
without moving a row and 59.1% are not.**

**The 40.8% is taken, and it is a second entry point rather than a change to
the kernel a decode step runs.** `packed_matmul_rows` gives one simdgroup a tile
of four consecutive rows instead of one output element: a lane loads a packed
byte, decodes its two codes, and multiplies them against the four rows of input
that named the same expert, so the weight row is walked once for four rows of
output rather than four times. A tile whose rows disagree about the expert walks
each row's own weight, which is what the untiled kernel does — so correctness
never rests on the caller having been right about an expert list, and a routed
bank tiled by mistake would be slow rather than wrong.

**Nothing about the order any product enters any sum moved, so the two kernels
agree bit for bit** rather than within a tolerance: an output element is still
one simdgroup's `simd_sum` over lanes walking the same bytes from the same
offset in the same stride. What moved is how many sums one load feeds.
`a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers` is where
that is held, over a shape whose tiles are one uniform, one straddling two
experts and one the call ends inside.

**Four rows and not eight, and the sweep is emphatic about it.** Wall time a
dispatch, over the shapes and lengths a prefill gives this kernel:

    rows a tile                  1        2        3        4        6        8
    q_proj, 385 tokens       607µs    463µs    402µs    381µs    538µs    893µs
    k_proj, 385 tokens       157µs    116µs    103µs     88µs    135µs    168µs
    shared gate/up, 385      611µs    459µs    411µs    365µs    551µs    837µs
    shared down, 385         650µs    470µs    415µs    360µs    508µs    509µs
    q_proj, 769 tokens      1214µs    933µs    814µs    788µs   1072µs   1759µs
    shared gate/up, 769     1225µs    927µs    842µs    814µs   1122µs   2087µs

Every shape turns at four and six is already worse than two. A tile holds a
running sum and an input offset a row, and past four of each the occupancy that
buys them costs more than the reads it saves — which is the same shape of
finding `dense_matmul`'s reduction width and this kernel's own lane width both
are, met a third time and turning much harder.

**What it bought is 1.45× of the bytes and 1.16× of the time, and the gap
between those two numbers is the next finding.** The matmul's rows at 769
tokens:

    kernel                calls    device     moved   achieved   of peak
    packed_matmul           121     3.72s   2476 GB   666 GB/s       81%
    packed_matmul_rows      336     2.05s    445 GB   217 GB/s       26%

against one row of 6.72 s and 4227 GB before it. The bytes came off exactly
where the arithmetic said they would — 4227 GB to 2921 is 1.447 where 40.8% cut
fourfold predicts 1.44 — and the time did not follow, because **the tiled rows
are no longer waiting on memory**. 217 GB/s against the untiled rows' 666 is a
kernel that has stopped being bandwidth-bound, and what it is bound by is
visible in the loop: four packed bytes to a lane are eight codes, and eight
codes against four rows are thirty-two input floats where untiled they are
eight. The input is small and warm, but it is now read four times as often per
byte of weight, and that is what the *column* tile below shares. Those four rows
are that commit's own; the tables under the column tile are what the same
dispatches cost today.

**And the 59.1% is taken too, by moving the rows rather than by tiling them.**
A token's six rows name six different experts, so the tile above can never reach
a routed bank whatever the prompt — but the rows *could* be laid out expert by
expert, and the selection that says how is already on the device, two dispatches
back. `group_by_expert` is a stable counting sort over it: 256 buckets, a
threadgroup atomic a bucket for the counts and a thread a bucket for the
placement, emitting where each row went and the expert list read through it.
**It is a permutation and nothing else** — `experts[i]` is `chosen[order[i]]` by
construction, so a token still reads exactly the six its router named, which is
asserted three ways rather than argued.

**The bank then reads through it at one end of each call and not both.**
`packed_matmul_grouped` is the same tile with an indirection: `gate` and `up`
gather, so their rows arrive in the router's order and leave in the grouping's,
and the activation between them and `down` inherit that layout for nothing;
`down` scatters, reading those grouped rows where they lie and writing each of
them back to the row the router named. So what the weighting behind the bank
reads is in the order it was always in, and nothing downstream knows the rows
were moved. The answer is the ungrouped dispatch's bit for bit, for the reason
the tile's own is.

**It costs a dispatch and no submission**, which is the constraint M8 left behind
about moving work near the router: the sort reads what the top-k wrote and writes
what the bank reads, so it goes between them in the command buffer a layer
already was. 1077 dispatches a prefill to 1117 at every length, and the
submissions where they were: 22 at 97 tokens, 42 at 385 and 43 at 769.

**What it bought, at 769 tokens and as the rows stood then:**

    kernel                    calls    device      moved   achieved   of peak
    packed_matmul_grouped       120     2.64s    1035 GB   393 GB/s       48%
    packed_matmul_rows          336     2.21s     445 GB   201 GB/s       25%
    group_by_expert              40    25.9ms       2 MB     0 GB/s        0%
    packed_matmul                 1    0.70ms     436 MB   624 GB/s       76%

against 3.72 s and 2476 GB of untiled routed banks before it — 1.41× of the time
and 2.39× of the declared bytes. **The declared figure is the worst layout the
shape allows and the truth is better than it**, which is the opposite bias to
everything else in this table and is deliberate: a grouped call's runs are as
long as the routing made them, a tile boundary falls inside a run far more often
than at the end of one, and a straddling tile walks each of its rows' own weight.
This side cannot count them — the expert each row named was never read back — so
`PackedBank::moves` charges one straddle per expert, which is never below what
the kernel reads. Measured against the layout the device actually produces, the
bound is 0.2% high at 97 tokens, 24% at 385 and 12% at 769.

**A token's bytes, which is the column that had to fall:** 3840, 3827 and 3825 MB
declared before, 3840, 2484 and 1951 after, and 3838, 2117 and 1813 once the
declared bound is replaced by the weight reads the kernel makes. **The 97-token
row is the finding and it is a negative one.** At 2.3 rows an expert the runs are
shorter than a tile, nearly every tile straddles, and a grouped call reads 581
weights where the untiled one reads 582 — so whatever a 97-token prefill gains
from being grouped, it is not bytes. It gains 0.69 s of device time to 0.59
anyway, because a grid of tiles is a quarter of the simdgroups doing four times
the work each, and that is worth something at a shape where the reads are not.

**And what compiling a third entry costs is worth stating, because it is not
nothing and it is not attributed.** Put `RUNS_A_GROUPING` past any prompt, so
that the entry compiles and no call reaches it, and a prefill takes 0.711, 2.91
and 6.17 s of device time against the 0.691, 2.91 and 6.01 of the commit before
this one — so 2 to 3% of a prefill is the price of the pipeline existing, before
any of it is used. `packed_matmul_rows`'s own
row carries most of that: 2.03 s to 2.23 at 769 tokens over the same 336 calls
and the same 445 GB. Why a pipeline that never runs slows one that does is
unexplained here; what is measured is that the grouping buys 1.06 s at that
length against it.

**The lever that was left was the input re-read, and it is taken.** Both tiled
rows sat far under the untiled kernel's 666 GB/s — 201 and 393 — and were 4.85 s
of a 5.11 s prefill's device time. Per output element a tile of four rows reads
the same `in_dim` input floats an untiled call does and a quarter of the weight
bytes, so the input was 32 bytes read for every byte of weight and the achieved
column was reporting that ratio rather than the memory. **A tile is four columns
wide as well as four rows deep now**: one lane loads a packed byte from each of
four consecutive weight rows and multiplies all four against the same two input
floats, so one read of the input serves four output columns and the ratio falls
to eight.

**It is the same three entry points and not a fourth**, which is what P3's price
for compiling a third one left this change owing. A column tile of one is the row
tile exactly,
so the column was written into the tile that was already there rather than
beside it — `packed_matmul_rows` and `packed_matmul_grouped` are the same two
entries with wider bodies, and the untiled kernel a decode step runs is
untouched. What that costs to compile is nothing measurable: the whole source is
380 to 500 microseconds at every width from one column to eight and the tiled
entry alone 124 to 155, with the widths ordered the wrong way round for the body
to be what decides it.

**What it bought, at 769 tokens:**

    kernel                    calls    device      moved   achieved   of peak
    packed_matmul_grouped       120     1.86s    1035 GB   556 GB/s       68%
    packed_matmul_rows          336     1.58s     445 GB   281 GB/s       34%
    fused_attention              42   548.6ms       1 GB     3 GB/s        0%
    group_by_expert              40    25.9ms       2 MB     0 GB/s        0%
    packed_matmul                 1    0.70ms     436 MB   624 GB/s       76%

against 2.64 s and 2.19 s for the same 120 and 336 calls before it, and at 385
tokens 548 GB/s to 739 and 213 to 290. **1.39 s came off the two tiled rows and
1.29 s off the prefill's device clock**, which is what says the change reached
the rows it was aimed at and paid for itself nowhere else: the two were 4.83 s of
5.11 and are 3.44 s of 3.82. The two rows big enough to have said otherwise did
not — `fused_attention` moved 548.9 ms to 548.6 and `dense_matmul` 38.63 to
38.64 — and `group_by_expert`, `moe_combine`, `swiglu`, both routers and the
untiled `packed_matmul` are each within 4% on rows of 26 ms and under. The two
that are not are `rms_norm` and `short_conv`, which moved 10.5 ms to 26.1 and
21.7 to 15.7 in opposite directions for 9.6 ms between them — 0.25% of a
prefill, on two of the smallest rows of a table that carries 8.4% of sampling
bias, and neither kernel was touched.

**Not one byte came off, and that is the point rather than a disappointment.**
Every output column is its own weight row, so the columns of a tile share no
weight byte and `PackedBank::moves` does not mention the width: 3840, 2484 and
1951 MB a token declared, and 3838, 2117 and 1813 once the declared bound is
replaced by the weight reads the kernel makes — the same six figures to the
megabyte either side. **This is the exact inverse of the row tile's finding**,
which took 1.447× of the bytes and 1.16× of the time. Bytes and time stopped
moving together the moment this kernel stopped being bandwidth-bound, and a
change that moves one without the other is what that looks like from each side
in turn.

**Four columns and not eight, and the sweep turns as hard as the height one
did.** Wall time a dispatch at the shipped four rows, over the shapes and lengths
a prefill gives this kernel:

    columns a tile               1        2        3        4        6        8
    q_proj, 385 tokens       410µs    361µs    324µs    282µs    334µs    493µs
    k_proj, 385 tokens        98µs     96µs     85µs     73µs     87µs    124µs
    shared gate/up, 385      425µs    365µs    336µs    292µs    359µs    486µs
    shared down, 385         392µs    352µs    290µs    254µs    292µs    496µs
    q_proj, 769 tokens       819µs    716µs    644µs    561µs    678µs    981µs
    shared gate/up, 769      875µs    735µs    678µs    583µs    714µs    973µs

Every shape turns at four and eight is slower than one column. **The rate this
sweep prints is over bytes that do not move with the width**, unlike the height
sweep's, which is what makes the columns comparable without an argument. A tile
carries a running sum per row *per column*, so four beside four rows is 32
accumulators a lane where the row tile alone wanted eight — but **that the turn
is register pressure is a reading rather than a measurement**: the widest
threadgroup the pipeline reports is the device's own 1024 at eight columns as at
one, which is the one place this side could have seen it.

**And the height did not move under it.** Re-swept at four columns, the rows a
tile turn at four on four of the six shapes and at three on the other two by
under 2%, which is where they turned at one column — so the two axes are
independent to the resolution this measures at, and `ROWS_A_TILE` is where the
row tile left it.

**M9's hypothesis did not hold, and the table is what says so.** It was that
every `(head, query)` threadgroup re-reading all keys is the next order of
magnitude here. `fused_attention` is 4.4% of a 385-token prefill's device time
and 7.5% of a 769-token one — growing with the prompt, as the hypothesis
implies, and two orders below where the time is. An attention kernel that cost
nothing at all would have taken 8.37 s to 7.82. Its share is **7.9% and 13.3%**
now that the matmul under it has shrunk twice, against 5.0% and 8.6% after the
grouping and 4.4% and 7.5% when the table was first taken — 548.6 ms of device
time at 769 tokens that has not moved by a millisecond across three changes to
the row above it. That is what a deferred finding looks like when everything
around it improves, and is the same thing that happened to `short_conv`; what
has not changed is that it is not where a prefill's seconds are, and what has is
that it is now the third-largest row rather than a rounding error.

**Nor is it the round trips, and this time that is measured rather than argued
from a submission count.** A prefill is one submission a layer — 42 of them at
385 tokens and 43 at 769, since one of its layers alone passes the bytes a merged
run may hold — and the wait on each is 97% and 98% execution, with `queued` at
nothing on every layer's row. So M16's pipelining reaches none of this: there is
no second command buffer behind the one being waited for, because a prefill never
merges two layers into one. The 42 submissions are 250 µs against a gap of seven
seconds, which is where that argument stood before, and the round-trip table now
says it rather than the arithmetic.

**Two things in the diagnosis are unexplained and are written down as such.** A
sampled prefill is *faster* in wall time than an unsampled one — 3.57 s against
4.62 and 4.65 at 385 tokens, 7.26 against 8.37 and 8.29 at 769 — while the
device's own clock does not move: 3.36 s against 3.32 and 3.36, and 6.96 s
against 6.96 and 6.87. It is not the first read of a length faulting its pages
in, because the two unsampled runs sit either side of the sampled one and agree
with each other and with the figure the command line reports cold. So about a
second of a prefill is this process's own, it is not execution, and putting each
dispatch in a compute pass of its own removes it. Nothing here has attributed
that, and no
number above rests on it — every row in the tables is device time. The other is
the pipeline that costs a prefill 2 to 3% by existing, above.

**A whole decoder layer is now one command buffer**, and twenty-six dispatches
on a layer that routes — twenty-seven where the prompt is long enough to lay the
routed rows out by expert first. Eleven are its attention: the input layernorm, the four
projections that read it, the two short convolutions behind the key and the
value, the two head norms over the query and the convolved key, the attention
step and `o_proj`. Three more are the two residual paths around the MLP — the
layer's two short convolutions, each of which adds the value its block began with
as a second addend where it writes rather than in a dispatch of its own, and the
second norm between them. The last twelve are the MLP: the router's gate, the
top-k over 256 sigmoid-corrected scores, each bank's gate, up, activation and
down, the softmax over the eight logits that selection named, and both banks'
rows weighted by it and summed. The twenty-seventh is the sort between the top-k
and the routed bank, which a decode step never dispatches: six rows over 256
experts group into runs of one and a tile of them shares nothing. Every value between them is a buffer the next
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
that follow it — 21.1 ms, where the 20.1 ms above is the same step at the
eight-token context every other measurement in this file is taken at:

    tokens in the block    1      2      3      4      6      9
    forward pass       23.3ms 31.1ms 38.6ms 43.1ms 62.5ms  79.5ms
    × a decode step      1.10   1.48   1.83   2.04   2.96    3.77
    submissions            15     15     15     15     15      15

    heads chained          1      2      3      4      6      8
    the chain           3.5ms  6.9ms 10.3ms 13.7ms 20.6ms  27.4ms
    × a decode step      0.16   0.32   0.49   0.65   0.97    1.29

**An extra token in the block costs 7.0 ms**, which is 0.33 of a decode step and
is exactly what the acceptance study measured — 10.5 ms against a 31.8 ms step.
Both the block and the step it is weighed against have fallen since, and by
enough of the same factor that the fraction is back where it started. Most of it
is the MoE and is fundamental — one token reads 6 routed experts a layer and nine
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

A head's guess costs 3.4 ms and reads 995 MB — its own 532 MiB, and `lm_head`
again to turn a hidden state into a token. **How that divides was inferred here
for two milestones, measured against the inference and found to disagree with
it, and has now been moved**: at six submissions a guess the device executed for
2.2 ms of 4.5 and the other 2.3 were the round trips and this process's own work
between them, where this file had 3.4 ms of bandwidth and 1.3 of round trip. At
one submission a head plus its `lm_head` it is 2.27 ms of execution inside 2.78
ms of wait. The study called the reference's per-head overhead "yours to win in
Rust"; mlx-vlm was near the old figure at 3.9 ms — on the 8-bit checkpoint, whose
heads are these heads byte for byte but whose `lm_head`, which a guess also
reads, its quantiser left in the original precision. **Only the `lm_head` half
of those bytes is the packed matmul's**, and reading four to a lane took that
dispatch 1.57 ms to 1.46 and the guess 4.85 to 4.70 — the same tenth of a
millisecond arriving twice, which is what says the head's own bfloat16 tensors
are the other half.

**And now where a chain's milliseconds go, which is a question this file had
answered for a decode step and a prefill and never once asked of the heads.**
The same tables, over the eight heads at one row, sampled:

    kernel            calls    device   share       moved   achieved  of peak
    dense_matmul         72   12.60ms   67.6%   4469.46 MB   355 GB/s     43%
    packed_matmul         8    5.26ms   28.2%   3489.14 MB   664 GB/s     81%
    rms_norm             32  342.10µs    1.8%      1.97 MB     6 GB/s      1%
    short_conv           32  277.45µs    1.5%      5.77 MB    21 GB/s      3%
    fused_attention       8  117.25µs    0.6%      1.05 MB     9 GB/s      1%
    swiglu                8   57.20µs    0.3%      3.15 MB    55 GB/s      7%

**A chain of eight heads reads 7.96 GB where a decode step reads 5.9**, and 4.5
of those are bfloat16 the quantisers never touched — so two thirds of the
chain's device time is the one kernel in the model reading a format nobody
packed. The two big rows are the same chain and the same bytes, and the packed
kernel beside it is at nearly twice the rate. That was the first half of
the answer and it is taken: four values to a lane, which is the lever the decode
step's own table above records this kernel taking twice. Over four alternating
pairs, every pair moving the same way and the two ranges not overlapping, **that
row was 161 GB/s against 205 and the chain's device clock 19.59 ms against
17.90**. The four rows under the two are 794 µs between them and three of them
are new: a head's norms, its convolutions and its activation are dispatches now
rather than loops on this side, where the same table charged this process 2.24
ms of `sconv`, `swiglu` and `rms_norm` rows for them and 26 µs of residual adds
the convolutions now carry as a second addend.

**The rates in that table are not the rates the one before it printed, and the
sampling is why.** These rows carry **+2.6% of asking** where the chain's first
table carried +60.6%: a timed dispatch is a compute pass of its own, and a chain
whose every dispatch was most of a command buffer paid for that far more heavily
than one whose nineteen share a buffer — 18.73 ms of sampled device time against
18.18 unsampled, where before it was 23.17 against 17.90. So `dense_matmul`'s
355 GB/s here and its 210 in the table this replaced are the same kernel reading
the same bytes at a different bias, and **the device's own unsampled clock is
what the comparison rests on: 18.18 ms against 17.90**. Why the bias is a
function of what shares a command buffer is not attributed here.

**The other half was round trips, and it is taken too.** A chain was **88
dispatches in 48 submissions — 1.8 a submission, against a decode step's 71.8**
— six a head, which were the five a partial handover took and the `lm_head` that
turns what it produced into a token, each committed and then waited for. It is
**160 dispatches in 16 submissions, 10.0 a submission**: one for the head, whose
input projection and whose eighteen dispatches of dense decoder layer are
nineteen in one command buffer, and one for the `lm_head` behind it. Of a 27 ms chain the device
executes for 18.2 and the wait is 22.4, where of the 43 ms chain before it the
device executed for 17.9 and the wait was 36.5:

    dispatches   a chain     waited  scheduled    queued   executed  unattributed
    1                  8     7.32ms   375.63µs  540.64µs     5.25ms        1.15ms
    19                 8    14.95ms   430.73µs  539.18µs    12.93ms        1.05ms

**`queued` is 1.1 ms across the whole chain where a decode step's run of layers
has 71 ms of it, and that is not what a merged head fixed.** M16's pipelining
still reaches none of this, and cannot: a head's guess has to *be* a token before
the head after it can embed it, so there is nothing behind the buffer being
waited for and there is no arrangement of these sixteen submissions in which
there would be. What the merge removed is the other 32 — the norms, the two
convolutions, the head norms, the activation and the residual adds each had a
kernel on the device already and what did not exist was the seam. It is the same
move the layers made four milestones ago, and what it needed was not a kernel:
`LayerProjections` and `DenseFfn` hold a weight either format answers for, so a
head's block is wrapped as the decoder layer it always was.

What is left after those two is this process's own: `sample` is 7.8% of the
chain and `readback` 2.1%, which are the argmax over 201024 logits and the
logits arriving to be argmaxed — eight times over, for the same reason `queued`
is nothing.

**And what is left of the chain is stated rather than attempted.** Of 27.4 ms
the device executes 18.2, and 12.6 of those are `dense_matmul` reading the 4.5
GB of bfloat16 the quantisers never touched — so the largest thing left in a
chain is not a round trip at all but the format the MTP shard ships in, which
`models/Inkling-Small-mxfp4/mtp.safetensors` carries verbatim from the BF16
original. **That is not taken here.** It changes what a head computes, where
merging its submissions cannot: no token can move, because the model verifies
every guess and a wrong guess costs a round its speedup and nothing else — but
*acceptance* can move, and acceptance is what the speedup is made of, so it
needs the heads' guesses held against the bfloat16 chain's before any timing
claim. Of the 4.2 ms of wait that is not execution, half is the eight `lm_head`
submissions, which a head could share a command buffer with if the model's own
final norm were on the device beside it.

**Speculation pays again at three depths where it paid at one.** Over 64 tokens
of a structured prompt, three passes round-robin over the depths so that a drift
moves them all, best pass each — and the whole sweep three times, so that every
figure here is the mean of three of those:

    k                      0      1      2      3      4
    ms/token           21.14  19.10  19.75  20.23  24.97
    tokens a round      1.000  1.829  2.560  3.048  3.368
    speedup             1.000  1.107  1.070  1.045  0.847
    accepted, by depth         85%  91/74% 84/74/63% 82/65/53/47%

**Those three sweeps are three alternating pairs against the six submissions a
head used to take, the order of the two halves flipped each pair**: 19.89,
21.05, 21.73 and 25.57 ms a token become the row above at `k` of 1, 2, 3 and 4.
Every pair moves the same way at every depth, and the two ranges are apart at
1, 2 and 3 —
19.88-19.89 against 19.06-19.13, 20.96-21.14 against 19.62-19.88, 21.35-22.03
against 20.13-20.41. **At `k = 4` they are not**, 25.30-25.93 against
24.03-25.47, so the depth speculation still loses at is the one depth this
milestone claims nothing about. The chain is 36.24 ms to 27.44 at eight heads
and 4.59 to 3.46 at one, neither range overlapping. Acceptance is untouched, to
three decimals, because the prompt and the model are the same.

**And the unspeculated step is where it was**, which is the constraint all of
this was done under: 21.13 ms against 21.16 over the same six readings, ranges
21.08-21.16 against 21.10-21.20. The two halves of that disagree about the last
0.05 ms — the run that speculates nothing reads 21.13 against 21.18 with its
ranges apart, and the `k = 0` row of the table above reads 21.13 against 21.14
with them across each other — which is what a change that touched no decode
dispatch and put a virtual call in front of 457 of them looks like when it is
measured at 0.2%.

**`k = 1` is what pays best and it pays 1.11×**, and `k = 2` and `k = 3` are
worth running again at 1.07× and 1.045× where they were 1.00× and 0.97×. What
changed is not acceptance and not the block: a `k = 1` round spends 3.5 of its
35 ms guessing where it spent 4.6 of 36, and a `k = 3` round 10.3 of its 61 ms
where it spent 13.9 of 66.

**A chain that cost nothing would put those three at 1.23×, 1.24× and 1.26×**,
against the 1.22×, 1.21× and 1.23× the same arithmetic puts them at over the
three pairs' own `before` half. So the ceiling is the workload's — acceptance
and what a block costs to verify decide it, and neither moved — and what still
separates `k = 1` from it is 0.12 where it was 0.15. This file said 1.28× when
it last stated that ceiling, off a one-head chain of 6.1 ms that three
alternating pairs now put at 4.59.

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

**And `k = 2` was a loss on this prompt when that sitting was taken**, at 26.87
against 22.07. It was never a win against the reference — 26.72 against 22.77 was
1.17× behind — but it was worth 1.10× against this engine's own unspeculated
step, and it was worth 0.82× against it there. Nothing about the speculation
changed — the same 128 tokens come out at `k = 1`, `2` and `4` as at `k = 0`,
byte for byte — and this prompt's first head is accepted 66% of the time against
the sweep prompt's 85%, which was never enough to pay for a chain of heads at
the step this engine had then.

**That sitting predates the merged head above and has not been retaken**, which
is what the `k = 2` row is worth reading as: the chain it paid for was 36 ms and
is 27, and what that comes to at 66% acceptance on this prompt is not a number
this file has, because a cross-engine claim needs both sides measured in one
sitting and the reference has not been run since. The `k = 0` row is the one the
merge cannot have moved, and the sweep above says it did not.

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
