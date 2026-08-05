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

`just test` is the one to run while iterating: **692 of the 766 tests, no
checkpoint, thirteen seconds.** Everything a fixture can settle is here — the
kernels against the CPU, the CPU against mlx-vlm's recorded activations, the
tokenizer against the whole vocabulary, the server against its own frames. The
38 that need weights report a skip and pass. It runs through libtest, which puts
a crate's tests in one process: opening a Metal device costs a second, so the 256
kernel tests are 13.6 s sharing a process and minutes with one each. Nothing in this
tier measures the process it runs in, which is what makes sharing one free.

`just test-full` is what has to pass at the ends of a series: **all 766 against a
real checkpoint, fifteen minutes.** The 692 the gated tier runs and the 74 the
timing tier does. The 54 gated tests — the 38 above and fifteen
of the measurements below, which need weights as well as a clock — are what
only weights can settle — that the packed tensors decode to what the reference
decodes, that 42 trained layers reproduce the recorded stack, that the engine
generates the oracle's own continuation, and that it generates the same
continuation while guessing four tokens ahead — and `--backend cpu` is the
oracle they are measured against, at 9.0 s a decoded token, which is where most of those
minutes go. This tier runs a process a test, which is what keeps a test that
bounds its resident set bounding only its own.

`just test-timing` is the seventy-four tests whose result *is* a number — a duration
they assert on, a resident set they bound, the three decode-step tables quoted
above, what a speculative round costs — run one at a time with nothing beside
them. **A measurement taken while fifteen other tests ran is a measurement of
the fifteen:** a round trip this repo has at
191 µs reports 598 under a parallel suite, and `.config/nextest.toml` records
what believing a number like that once cost. `#[ignore]` is what keeps them out
of the two runs above, and what selects them here.

**Which of the three a commit needs is not "all of them, every time".** `just
test` after every edit; `just test-timing` for anything whose result is a number;
`just test-full` before the first commit of a series and again before the last.
Most of `test-full` is the CPU oracle at 9.0 s a decoded token, and that oracle
cannot have changed between two commits that never touched the CPU path — so
running it per commit is a series of six paying twenty minutes to re-prove the
same thing five times. A commit that touches no `.rs` file needs none of the
three; the pre-commit hooks already skip clippy on those by config.

### Measuring two refs against each other

    just bench HEAD~1 HEAD decode
    just bench HEAD~1 .    decode --context 8192
    just bench HEAD~1 .    prefill --tokens 769
    just bench v1 v2       sweep --depth 4
    just bench-engines                        # this engine against mlx-vlm
    just bench-session                        # a conversation, kept against not

**A decode step is the one measurement here with a context to be taken at**, and
until the occupancy turn wanted one it was always taken at the structured
prompt's own 34 keys — which is the one length nobody has, on the one row of this
file that grows with the context. `--context` tiles the prompt to the length
asked for first. The other three refuse the flag rather than drop it: a prefill's
context is its prompt and `--tokens` already says how long that is, and a sweep
and a cross-engine table fix their own prompts because acceptance is the
workload's.

Every timed claim in this file is paired and alternating — build A, build B, run
them in one sitting with the order flipped each pair, and report whether the
ranges overlap — and what that discipline used to cost was a checkout and a
Metal-crate rebuild per flip, up to fourteen of them for one figure. **The
rebuild bought nothing**: the two binaries do not change between pairs. So each
ref is built once into `target/bench/bin/<sha>` and kept, and the pairs are
process launches against binaries that already exist. `.` is the working tree,
which is the arm a change is measured from before it is a commit at all.

**And one of them measures more than one request.** Everything else here is a
call — a prefill, a decode step, a prompt and its answer — and a cache kept
between requests is worth nothing on any of them, because none of them has a
between. `just bench-session` is a conversation instead: several turns, each
adding a question and each answered. See "What a conversation costs when its
cache is kept".

The four things it measures are the four this file quotes: a decode step, a
prefill at a given length, the end-to-end `k` sweep with its acceptance and its
speedups — the last of which are divided against *that run's* own `k = 0`,
because a sweep whose speedup row comes from another sitting carries the drift
between the two — and what a prompt and its answer cost together. What comes back
is the per-arm mean, both ranges, whether they overlap, and how many pairs moved
the way the means did, which is this file's own standard for an effect. It runs
one arm at a time and opens one Metal device apiece, for the reason
`.config/nextest.toml` gives.

**And the last of the four takes the other engine as its second arm.** Nothing in
the protocol between the harness and an arm says which engine printed the
readings, so `just bench-engines` puts mlx-vlm on one side and this engine on the
other and alternates them the same way — which is what a cross-engine claim needs
and what running one engine and then the other cannot give, since this host has
drifted 1.7% inside a single sitting. See "Against the reference, end to end".

Text in, text out, streamed to stdout as each token is decoded:

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' -n 4

A decode step is about 16.3 ms — 14.6 ms a token speculating three deep, which is
68.3 tokens a second — and the timings go to stderr so stdout stays pipeable.
**Both of those are taken at an eight-token context and neither is what a user
feels**; "Against the reference, end to end" and "Where a decode step goes as the
context grows" below are the figures that are, and they do not read the same way
at every prompt length — 18.1 ms a token at a 97-token context and 26.9 at
8192. The prompt reaches the tokenizer as it stands,
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

### Three numerics behind one flag, and which of them is checkable

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' \
        --numerics production

**`--numerics` is `reference` unless a command line says otherwise, and nothing
in this file is measured under anything else unless it says so.** Every kernel
here has been held to one standard since the first — the answer is the CPU
path's bit for bit, and the recorded continuation `[656, 13, 623, 180069, 86333,
60500, 220, 23]` has never moved — and that standard is what makes a mutation
falsifiable. It is also a ceiling, and "Whether the fast structure can keep the
bits" is where this file first said which one: a hardware `simdgroup_matrix`
multiply-accumulate sums its `k` dimension in an order the instruction defines
and this side does not choose, so **any kernel built on one is ruled out before
it is written**. The flag is what lets it be written and measured anyway.

**Neither `reference` nor `production` is "more accurate", and saying so is not
a hedge.** Both sum the same exact products of the same exactly-decoded codes —
MXFP4's sixteen values are one table and a group scale is a power of two, so no
product either path forms is rounded at all. What separates them is the order the
sums are taken in and nothing else, and a matrix instruction's order is not the
worse of the two; on a long reduction it is usually a little better. What the
reference has that the production path cannot have is an **oracle**: an order
this side picked, that `--backend cpu` reproduces exactly, so that a wrong token
has a witness. That is the whole of the difference and it is a difference about
checkability rather than about precision.

**`rounded` is the third word and it is the one that does give something up.**
It stages the packed matmul's two tiles as 16-bit floats, so `element × scale`
carries eleven bits of significand where every other path here carries
twenty-four: the sentence above stops being true of it, which is exactly why it
is a third word rather than a faster `production`. **Nothing about what
`production` guarantees moved to make room for it** — the two words behind the
flag compile the same two entries at the same block shape through the same
predicates, and `the_production_source_stages_and_multiplies_in_full_width` is
where that is held. It is not a default, it is not what anything in this file is
measured under unless the row says so, and what it is worth is under "The operand
door, opened as a third word": **6.7 to 7.3% of a prefill's device time, and 384
of 384 sampled argmaxes unmoved.** Whether that trade is one this engine should
make is not a question this file answers.

**So the chain a disagreement is bisected through gains a link.** It was
CPU → Metal, one arrow, settled by rerunning a command with `--backend cpu`. It
is now **CPU → Metal under the reference → Metal under the production
numerics**, and the middle of the three is what says whether a wrong token came
from the kernel structure or from the arithmetic inside it. `--backend cpu` plays
the same role it always did; the new arrow is the one `just diverge` walks.

**The flag selects the innermost compute and nothing else.** Tiling decisions,
the submission structure, the grouping, KV handling, `splits_for`, both occupancy
turns — all shared, all exercised whichever way it reads. A kernel behind this
flag is a different accumulation over the same dispatch, taking the same bindings
from the same encoder, at the same shapes the same predicates chose. That bound
is the point rather than a tidiness preference: `attention.rs` and `matmul.rs`
are the two most-edited files here, and a fork of the engine at any level above
the accumulate would have two of everything that moved in the last four
milestones and would rot inside two more.

**Nothing changes for a caller who does not ask.** Under the reference the
production entries are not compiled, not dispatched and not in the pipeline
cache — `PackedMatmul::new` builds exactly the three kernels it built before this
flag existed. `--numerics` on `--backend cpu` is refused rather than dropped: the
CPU path has one arithmetic, and a run that took the word and ignored it would
print a command line saying something other than what it did.

    just diverge                                     # the corpus through the reference and production
    just diverge --against rounded                   # and through the reference and the operand word
    just bench-numerics reference production prefill --tokens 2048
    just bench-numerics production rounded  prefill --tokens 2048
    INKLINGRS_NUMERICS=production just test-timing   # the per-kernel table on the other side

**What is behind the flag today is the packed matmul's two tiled entries and the
attention step's block of query rows**, which are the two kernels the profile's
own share column puts at 96.7% of a long prefill's passes between them. What the matmul's are worth is under "What
the matmul costs on the other side of the line": 2.85× on the two matmul rows at
16384 tokens, 37 to 45% off a prefill's wall, nothing at all on a decode step.
What the attention block is worth is under "What the attention step costs on the
other side of the line": **19.4× on the two attention rows at 16384 tokens**, a
prefill of that length at 33.33 s against 109.11, and again nothing at all on a
decode step. **No token has moved over 384 sampled argmaxes with both of them
behind the flag.**

**A decode step reaches neither, and D5 is where that stopped being a fact about
the predicates and became one the instrument states.** Both entries are given a
call only where its rows are two blocks' worth, so a corpus of long enough
prompts reached everything the flag selects and a prompt-length floor was a
sufficient guard. An entry a *decode step* dispatches would be gated on nothing —
a call of one row reaches it — so no length could say whether a differential run
had ever run it. `just diverge` now holds what it *dispatched* against the list
each kernel keeps of what it put behind the flag, and prints which entries the
flag selected at a prefill and at a decode step. See "The walk a decode step
could stop doing serially", where the one candidate for that second list was
built, measured at 31% the wrong way, and taken off the path.

**It is not the default and the recommendation is still that it should not be**,
and the reasoning is in that second section, kept apart from the numbers — but it
is a closer question than it was, because what is now on the other side of the
line is most of the engine rather than one half of it.

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

    submit and wait      28    65%      of which the device executed for 18 ms
    dispatch encode    1080    29%
    readback              1     0%
    everything else                     5%

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
**93%** of the step, which is a share above the row it sits in: a run of layers
commits part way through and keeps encoding, so a command buffer executes while
this process is charging its time to `dispatch encode`. Nothing an operation of a
layer would open a scope around is left in the table at all: what remains beside
the round trip is encoding it and the embedding at the start. **And the back of
the model has left it too, to the last operation** — the final norm, the muP
divide and the argmax over the vocabulary were the last three rows here that were
arithmetic rather than asking, and all three are dispatches: there is no `sample`
row, and `readback` is three microseconds where it was seventy, because what a
step reads back is a token rather than the 200058 logits it was taken from. See
"The tail of a step" and "Sampling on the device" below.

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
The head's submission was the other kind: 1.98 ms of wait around 0.66 ms of
work, so 1.3 ms of it bought nothing, and that was the price of the seam that
read the stack's rows back to norm them on this side. **That seam is gone and
the row with it** — see "The tail of a step", where the same row is re-measured
at 0.98 ms of wait around 0.66 ms of work before it is removed, because a figure
taken three milestones ago against a step that has changed twice is not a figure
about this step.

**What was not in that table is the encode, and that was the finding.** A command
buffer executes nothing until it is committed, and a step committed after
encoding all 1076 of the stack's dispatches — so its 4.4 ms `dispatch encode` row
was 4.4 ms with the GPU idle, ahead of the wait rather than inside it. **So the
run commits at the first layer boundary past 64 dispatches now and carries on
encoding into the next buffer**, waiting for none of them until somebody reads
the rows. A MoE layer is 26 dispatches, so that boundary is three of them, and
the same table reads:

    dispatches   a step     waited  scheduled    queued   executed  unattributed
    54                1     1.51ms   141.12µs   10.98ms     1.51ms        0.00ns
    78               12    11.17ms     1.08ms   70.83ms    14.97ms        0.00ns
    88                1     1.17µs   120.12µs  467.13µs     1.55ms        0.00ns

**A `queued` column of 71 ms inside a 20 ms step is the whole of what changed.**
Twelve command buffers sit in the queue while the ones ahead of them run, and the
one this process blocked for 1.17 microseconds is the last of them — committed
and finished before there was anything else to wait for. `unattributed` is
nothing on any row, because the three parts of a submission a run committed now
account for more than the wait rather than less. **There is no one-dispatch row
here any more**: the 54 is the last two layers' 52 dispatches with the final
norm and `lm_head` behind them, where the head used to open a buffer of its
own.

Over seven alternating pairs, every pair moving the same way and the two ranges
not overlapping, a decode step is **26.34 ms to 20.11**. **The device's own clock
did not move** — 18.13 ms against 18.10 — so all 6.2 ms of it is this process's
wait and none is work the GPU stopped doing, which is what says the change is
scheduling and nothing else. A second seven pairs taken before the device's clock
was read beside the step put the same figures at 26.39 and 20.18. The recorded
continuation did not change, and neither did the peak resident set.

**And now which kernel owns which of those 18 milliseconds.** The device
timestamps a command buffer, and a decode step is fourteen of them around 1078
dispatches, so until this landed that figure was one number with nine kernels
behind it. It is now nine numbers, each beside the bytes that dispatch said it
moves and what that comes to against this machine's 819 GB/s:

    kernel            calls    device   share       moved   achieved  of peak
    packed_matmul       457   15.07ms   70.4%   5932.36 MB   394 GB/s     48%
    rms_norm            169    1.90ms    8.9%      5.94 MB     3 GB/s      0%
    short_conv          168    1.38ms    6.4%     22.02 MB    16 GB/s      2%
    fused_attention      42    891µs     4.2%      5.62 MB     6 GB/s      1%
    router_top_k         40    549µs     2.6%      0.08 MB     0 GB/s      0%
    dense_matmul         40    466µs     2.2%     85.24 MB   183 GB/s     22%
    swiglu               82    429µs     2.0%      8.26 MB    19 GB/s      2%
    router_weights       40    380µs     1.8%      0.00 MB     0 GB/s      0%
    moe_combine          40    335µs     1.6%      5.90 MB    18 GB/s      2%

**That table is a true table about a context nobody has, and saying so is this
milestone's finding.** It is taken over the eight-token prompt the activation
capture recorded, as every decode figure in this file was until
`which_kernels_own_a_decode_step_at_each_context` took the same table at 97, 385
and 769 keys. Exactly one row moves with the context and the rest are flat inside
20%:

    kernel                97 keys   385 keys   769 keys
    packed_matmul         16.55ms    15.68ms    15.22ms
    fused_attention        3.93ms    12.50ms    17.35ms
    rms_norm               2.05ms     1.93ms     1.88ms
    short_conv             1.64ms     2.00ms     2.29ms
    router_top_k           614µs      577µs      559µs
    dense_matmul           487µs      469µs      457µs
    swiglu                 454µs      436µs      420µs
    router_weights         414µs      397µs      380µs
    moe_combine            366µs      345µs      343µs

The matmul reads the model once whatever the context, so its 5932.36 MB is the
same figure in all three columns. **`fused_attention` is the whole of the
growth** — 13.4 ms of it against a step that grew by 14.0 — and the paragraph
below is what was done about it. Those rows carry the sampling bias the last
line of the table names, which is why they are read against each other rather
than quoted absolutely.

**The packed matmul is 70% of the device's time and it is the only kernel here
doing bandwidth's work.** Its 5.9 GB is what the checkpoint's shapes say a token
reads — six of each MoE layer's 256 experts and both shared ones, plus every
layer's own projections — arrived at from a dispatch's own declaration rather
than from that arithmetic, and the two agree. 48% of the machine is what a lane
holding four packed bytes rather than one is worth, against 34% before it — and
394 GB/s is four fifths of the way up the 284-to-424 GB/s M2's isolated matmul
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
    over     8 keys                      10.16µs   1.43µs             86%
    over    97 keys                      31.89µs   1.89µs             94%
    over   512 keys                      59.74µs   1.59µs             97%
    over  4096 keys                     409.31µs   1.69µs            100%

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

**`fused_attention` was waiting on itself, and it took a table at three contexts
to see it.** The launch is under 2% of it past a hundred keys and the rest tracked
the span — but so did the *rate*, which is what says the span was not the
problem. This kernel gave one threadgroup to each of 32 query heads, so a decode
step ran 32 threadgroups on a machine with 80 cores: 48 idle, and the 32 that
were not had one threadgroup apiece with nothing to interleave against, on a
loop whose every 32-key tile is four barriers and a dependent read. It held 9 to
16 GB/s against this machine's 819 from 97 keys to 65536 — a kernel that is
neither near the memory nor getting further from it, which is the signature of a
dependent chain rather than of a bandwidth.

**So the span is cut across threadgroups and the two halves are folded.** A
threadgroup takes one split of one query and leaves an unnormalised weighted sum
beside the peak it is relative to; `attention_combine` takes the largest peak,
rescales every split onto it and normalises. That is the lever "Sampling on the
device" below records the argmax taking, for the same reason and against the same
finding: one threadgroup over a row of the vocabulary was this process's own
argmax to within its spread, and the whole of what a device argmax was worth
turned out to be the cut. `rms_norm` measured the same arithmetic and declined
it, which is why this is a number rather than a rule — a split costs a dispatch,
so it has to buy more than one.

**The cut is over the whole span on tile boundaries and not over the live
range**, which is what keeps the loop bound above exact: a kernel with the bound
removed walks the same tiles, so a split holding no live key leaves a peak of
`-1e30` or `-INFINITY` that the fold rescales by a zero that is exact.
`the_bounded_loop_is_the_unbounded_one_bit_for_bit` is unchanged and is now
driven through a 32-way split rather than, as it had been, through none.

**What it is worth, on the 42 dispatches the stack runs**, one query at both
kinds of layer:

    keys                      97      385      769     2048     8192    32768
    before               2.81ms  10.09ms  14.32ms  18.96ms  40.14ms 145.38ms
    after                1.22ms   2.28ms   3.87ms   7.19ms  16.06ms  28.58ms
    after, refitted      1.34ms   2.46ms   4.33ms      —        —        —

The per-key cost of a global layer went **0.588 µs to 0.076**, and it holds that
figure from 769 keys to 65536 — so the row is still linear in the context and the
slope is 7.7 times shallower. The windowed layers were already flat past their
window and are now flat lower.

**And one split is the kernel that was there**, which is what lets a prefill be
untouched: `splits_for` gives 1 wherever the grid already fills the machine — a
769-token prefill is 24608 threadgroups — so no fold is encoded, nothing is
allocated for one, and the dispatch is the same arithmetic over the same tiles
writing the same bits.

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
them: the rows above sum to 21.4 ms against the 18.2 ms the same reading puts an
unsampled step's device time at, so each carries a couple of microseconds it
would not have — 18% across the table, and more of it on the short rows than on
the long one. **The bias grew in both denominators and why is unmeasured**: 3.2
ms of over-reporting where it was 2.7, over about the same number of boundaries,
so a boundary costing what it costs whatever is inside it does not explain it. The
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
MLP then runs — 1122 dispatches, and the only thing that ends a command buffer
is a run reaching the dispatches it commits at or somebody reading the logits. **What those have in common is that a seam had to be able to express
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
`packed_matmul_grouped` one**, in the same dispatches, submissions and 953
buffers of 17.6 MiB the commit before it ran. (Those two counts were 1077 and 15
when that pair was taken and are 1078 and 14 now, for a reason that has nothing
to do with the tile — see "The tail of a step".) The recorded continuation
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
submissions where they were: 22 at 97 tokens, 42 at 385 and 43 at 769. (The tail
has since taken 97 tokens to 1118 dispatches in 21 submissions and left the other
two where they are — see "The tail of a step".)

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

**M9's hypothesis has been re-measured and it is where the work is now — but it
is a decode finding rather than a prefill one.** `fused_attention` is 548.47 ms
of a 769-token prefill — 13.3% of the passes the profile sums and 14.4% of the
3.80 s the command buffers clocked — and an attention kernel
that cost nothing at all would take that prefill's 5.39 s to 4.84. That is what
the paragraph below always said and it still holds. **What changed is the other
regime**: the same span-walking was 19 ms of every decode token at a 769-token
context, which over 128 generated tokens is 2.4 s — larger than the whole
`packed_matmul_grouped` row. **That is where the work then went, and it is 4.3 ms
now** — see the split under "And now which kernel owns which of those 18
milliseconds". The kernel this milestone should have been about was this one, and
the measurement that said so was the cross-engine table rather than the prefill
profile.

**M9's hypothesis did not hold against a prefill, and the table is what says so.** It was that
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

**Two things in the diagnosis were unexplained, and both were re-checked rather
than carried.** A sampled prefill is still *faster* in wall time than an
unsampled one — 4.05 s against 5.39 and 5.36 at 769 tokens — while the device's
own clock does not move: 3.80 s against 3.81 and 3.80. **So it is not gone**, and
S5's removal of the host sampling round trip did not touch it: putting each
dispatch in a compute pass of its own still takes 1.3 s off a prefill's wall and
nothing off its execution. What that number is worth reading beside is the
prefill itself — **1.58 s of a 5.39 s prefill is not device time at all**, which
is 29% of it and larger than any kernel row below the first two. The other
unexplained figure, the pipeline that costs a prefill 2 to 3% by existing, is
where it was.

### Why the two tiled rows report bandwidths a factor of two apart

**This was the largest unexplained number in the prefill table and it is now
explained.** `packed_matmul_rows` reports 283 GB/s where `packed_matmul_grouped`
reports 556 — this milestone's own re-reading of the two rows the table above
records at 281 and 556 — at the same tile shape and out of the same source
string — the two
entries are one string with three expressions substituted and nothing else. The
two things that could separate them are the indirection and the weight a dispatch
walks, and both were crossed rather than argued about: one shape held fixed at a
769-token routed bank's, over banks of 1 to 256 experts, through both entries
with the rows pre-sorted so the grouping is the identity and the tiles are the
same tiles.

    experts        a run   distinct    untiled       tiled     grouped        tiled      grouped
    1               4614       4 MB     7829µs      3501µs      3556µs    1469 GB/s    1446 GB/s
    4               1153      18 MB     7845µs      3497µs      3551µs    1471 GB/s    1460 GB/s
    16               288      71 MB     7837µs      3530µs      3585µs    1457 GB/s    1491 GB/s
    64                72     285 MB     7818µs      3635µs      3732µs    1415 GB/s    1604 GB/s
    256               18    1141 MB     7672µs      4015µs      4277µs    1281 GB/s    1999 GB/s

**Neither candidate is worth a factor of two.** The indirection is 1.6% at one
expert and 6.5% at 256, the second of which includes the sort dispatch. The
weight a dispatch walks is 15% across a 285-fold change in it — 4 MB to 1141 MB
for 3501 µs to 4015. **But the two rate columns part by 1.56× over that same
range while the two time columns stay within 6.5%**, and that is the whole of the
finding: the rates are moving because their denominators are, not because the
kernel is.

**`achieved` is `PackedBank::moves` over device time, and `moves` charges a whole
weight per tile.** What each of a prefill's nine packed shapes declares against
the weight it actually holds:

    shape                 rows  distinct  declared    over    a call    achieved   a prefill
    q_proj, o_proj         769      9 MB   1720 MB    ×193    1124µs   1531 GB/s     94.40ms
    k_proj, v_proj         769      2 MB    430 MB    ×193     297µs   1449 GB/s     24.93ms
    r_proj                 769      1 MB    215 MB    ×193     148µs   1452 GB/s      6.22ms
    shared gate, up       1538      9 MB   1716 MB    ×192    1170µs   1466 GB/s     93.64ms
    shared down           1538      9 MB   1716 MB    ×192    1009µs   1701 GB/s     40.36ms
    dense gate, up         769     36 MB   6881 MB    ×193    4287µs   1605 GB/s     17.15ms
    dense down             769     36 MB   6881 MB    ×193    6376µs   1079 GB/s     12.75ms
    routed gate, up       4614   1141 MB   8552 MB      ×7    4292µs   1992 GB/s    343.38ms
    routed down           4614   1141 MB   8552 MB      ×7    3977µs   2151 GB/s    159.06ms

**Every shape on the tiled entry is charged 193 times the weight it holds and a
routed bank is charged 7.** Across a whole prefill that is 444765 MB declared
against 2.34 GB of distinct weight for `packed_matmul_rows` — ×190 — and 1034803
MB against 129 GB for `packed_matmul_grouped` — ×8. **The two rows divide by
denominators inflated against each other by a factor of 24**, so 283 and 556 are
two amplification factors rather than two bandwidths, and the ratio between them
is mostly the ratio of those amplifications.

**Which is what says neither row is near this machine and the table's `of peak`
column is misleading for both.** The bytes that *must* come from memory are the
distinct ones: 2.3 GB over 1.57 s for the tiled row, which is 1.5 GB/s, and 129
GB over 1.86 s for the grouped one, which is 69 GB/s. Against 819. The rates
above 819 in both tables are the arithmetic saying the same thing from the other
side — a dispatch cannot move 8552 MB in 4292 µs, so most of what it is charged
for it did not fetch.

**So grouping does not help and rows is not broken, and there is no factor of two
of headroom here to go and get.** That is a negative finding and it is the point
of having taken it: the number that looked like the largest deficit in the
prefill table was an artefact of the denominator, and a milestone spent chasing
it would have found nothing.

### The one lever this left, measured and declined

**`ROWS_A_TILE` was fitted on a bank small enough to stay in cache**, which is
the one thing its own sweep could not ask about: at one expert and 2 to 36 MB
every weight read a taller tile saves was already a cache hit, so the sweep turns
at four and says so emphatically. A routed bank is 1141 MB a dispatch. Swept
there instead:

    rows a tile         a call  declared    achieved   a prefill
    2                   4549µs  11417 MB   2510 GB/s    545.83ms
    3                   4115µs   9127 MB   2218 GB/s    493.85ms
    4                   4243µs   8552 MB   2015 GB/s    509.21ms
    6                   5246µs   9109 MB   1736 GB/s    629.49ms
    8                   6935µs  10526 MB   1518 GB/s    832.18ms

**The turn moves from four to three and is worth 3.0%** — 509 ms of a prefill's
routed dispatches against 494, which is 15 ms of a 5.39 s prefill, 0.3%. **It is
not being taken, and the reason is the other regime.** `tiles` refuses a run
shorter than the height, so the height is what keeps every shape a decode step
and a speculative round dispatch off the tiled entry — and a round of depth `k`
verifies `k + 1` rows in one pass, so `k = 2` dispatches a three-row projection
that is tiled the moment the height is three. That is the 16.48 ms figure and
60.7 tokens a second, risked for 0.3% of a prefill. `a_calls_rows_share_a_weight
_read_only_where_they_name_one_expert` now asserts it directly: every block a
round can propose stays untiled at this height, and a tile of three would not.
**The height may rise and may not fall**, and nothing here moved it.

**What was changed in the kernel: nothing.** The two entries, the tile shape, the
grouping and the predicate that selects between them are where P4 left them. What
was added is three measurements and one assertion, because what the milestone
found is that its premise was wrong — see "Against the reference, end to end" for
where the time actually is, and `fused_attention` below for the row that Part 1
promoted from a rounding error to the largest single item this engine has.

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
measurement.** A decode step allocates 20 to 34 MiB across the 997 buffers its one
run retains, so the 160 MiB the budget allows is about nine rows of this stack —
which is the deepest block the eight heads can ask for, and the width the table
under "Speculating with the MTP heads" still submits in fourteen, the same as a
single row. It is also why the budget does
not reach a prefill: ten tokens already pass it, so every prompt worth the name
is a submission a layer, exactly as it was.

**So a decode step became two submissions**, one for the forty-two layers and one
for the head, where it was 43 and 87 and 249 — and is fourteen now for a reason
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

## The operand door, opened as a third word, and prefill's ordering

**The door T3 left open is now two numbers, and the first is that no token
moves.** 16-bit operands are built here as a **third word of `--numerics`**
rather than as a change to what `production` does, and A7's divergence corpus
through them reads **384 of 384 sampled argmaxes identical, no prompt parted**.
Every reordering change in this repo has come back 384/384; operand rounding is a
much larger perturbation and comes back 384/384 too.

**The second is that it is worth 6.7 to 7.3% of a prefill's device time**, paired
and alternating at 2048, 4096 and 8192, every pair the same way and the ranges
apart at all three, against a null pair that read 0.0%. That is eight times what
both of T3's levers returned between them, and it takes the routed bank from
**62.5% of the instruction ceiling to 66.7%** — the largest move on this kernel
since the block itself.

**It is not a default and it is not a recommendation.** What it costs is the
sentence `--numerics production` is documented on, and that sentence is untouched:
`reference` is still bit-exact and still the default, `production` still forms
exact products and still differs from the reference by summation order alone.
**The decision is the owner's and this milestone does not take it.**

**And prefill's ordering turns out to be a question the tree had already
answered.** D4's concurrent encoder is not gated on a shape — it has reached
every prefill since the day it landed — and what it finds there is a sequence as
narrow as a decode step's: **501 of a prefill's 665 groups are one dispatch**,
against 546 of a decode step's 670. It is worth **1.4% of a 2048-token prefill's
device time and nothing measurable at 8192**. That closes it, with the
distribution rather than with an argument.

### The divergence, which is what decides it

**Reported first, because a speed figure read before it would be an argument
rather than a measurement.**

    just diverge                     # the reference against production
    just diverge --against rounded   # the reference against the operand word

    six prompts, six distributions, 64 to 2123 tokens, 64 argmaxes each
    the reference against production      384 of 384 agreed, no prompt parted
    the reference against rounded         384 of 384 agreed, no prompt parted

**And the run says it reached the arithmetic rather than saying the harness
works.** D5's dispatch-list differential holds what each arm *dispatched* against
the list each kernel keeps of what it put behind the flag. Under `rounded` the
corpus selected `mma_attention`, `mma_matmul_grouped` and `mma_matmul_rows` at a
prefill and **nothing at a decode step** — the same three `production` selects,
which is what says the two words are the same entries at the same shapes rather
than two different coverings of the model.

**The recorded continuation is unmoved under all three words.**
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` off the device under `reference`,
`production` and `rounded` alike, and off `--backend cpu`.

**This was the likely outcome, and that is a reason to have measured it rather
than a reason to have skipped it.** S3 put the narrowest first-to-second logit
margin in the corpus at 1.56e-2, about 75× this drift, so the arithmetic said the
tokens should hold. The arithmetic in this file has also said the matmul was
bandwidth-bound, that two ablation arms bounded two levers, and that a barrier's
fixed cost predicted a *slower* step. **384 agreeing argmaxes is a measurement
and 75× is a hope.**

### What the word costs the answer, and the shape of it

Against an f64 accumulation of the same products, at the reductions
`a_block_answers_the_reference_tile_where_neither_extent_divides_it` uses:

    a reduction      the reference       the block  rounded operands
    32                      8.3e-8          1.3e-7            1.8e-4
    128                     1.4e-7          3.5e-7            1.7e-4
    512                     8.4e-8          6.6e-7            1.5e-4
    2048                    1.1e-7          1.5e-6            2.1e-4
    4096                    1.0e-7          1.6e-6            1.8e-4

**Two decades above the block's own, and the size is the smaller half of it.**
The block's drift *grows* with the reduction — twelvefold across these five —
because a fragment accumulator carries the whole reduction as one running sum. A
drift that is flat is not a chain at all: it is the operand, arriving already
rounded and staying that far off however few products are summed. **That shape is
what separates "this word is less accurate" from "this word has a bug"**, and it
is asserted rather than only printed.
`what_the_operand_word_costs_the_answer_is_flat_in_the_reduction` runs in the tier
that runs after every edit, and it fails both if the drift lands *inside* the
block's own bound — a word whose staging never narrowed, which would compile,
dispatch, agree with everything and not be what it says it is — and if it grows
like a chain.

### What the operand width is worth

**The width reaches the traffic under one fragment read and not the other**,
which is T3's finding and is why the word ships behind the read T3 shipped. A
lane pulling the two elements it owns pulls four bytes at half where it pulls
eight at float; a `simdgroup_load` is one instruction over the whole fragment
either way. Each column is against the block compiled with its own read, warm and
swept both ways, and the padding is swept because
[`MMA_STAGED_PADDING`]'s bank argument is derived against a *four*-byte staged
element and does not separate these three at two:

    a fragment read through simdgroup_load
    at                              the block   rounded at 4      at 8     at 12
    2048   q_proj, tiled               3.79ms            +9%       +9%       +5%
    2048   a routed bank, grouped     16.79ms            +8%       +9%       +5%
    4096   q_proj, tiled               7.56ms            +9%       +9%       +5%
    4096   a routed bank, grouped     26.37ms            +8%       +8%       +4%
    8192   q_proj, tiled              15.08ms            +9%       +9%       +5%
    8192   a routed bank, grouped     52.48ms            +8%       +8%       +5%

    a fragment read per lane, which is the shipped one
    at                              the block   rounded at 4      at 8     at 12
    2048   q_proj, tiled               3.76ms            -2%       -2%       -8%
    2048   a routed bank, grouped     16.63ms            +0%       +1%       -7%
    4096   q_proj, tiled               7.50ms            -1%       -1%       -8%
    4096   a routed bank, grouped     26.14ms            -0%       -1%       -6%
    8192   q_proj, tiled              14.97ms            -1%       -1%       -8%
    8192   a routed bank, grouped     52.16ms            -1%       -1%       -6%

**12 is what ships**, and it is set from that sweep rather than from the bank
arithmetic — 4, 8 and 12 all put a fragment's eight rows on eight distinct banks
at two bytes, so the argument that set 4 does not choose between them and the
clock does.

**And on a whole prefill**, one build under the two words, seven alternating
pairs each, the order flipped every pair:

    tokens         device          wall        of the instruction, routed bank
     2048   3231.8 -> 2997.0   5558.1 -> 5229.7      62% -> 66%   (49.0 -> 52.7)
     4096   5995.5 -> 5583.1   9050.3 -> 8526.2      62% -> 66%   (62.2 -> 66.5)
     8192  11715.5 -> 10934.6 16735.7 -> 15958.7     62% -> 67%   (62.5 -> 66.7)

    device  -7.3%  -6.9%  -6.7%    7 of 7 apart at every length
    wall    -5.9%  -5.8%  -4.6%    7 of 7 apart at every length

**Medians and means agree to a tenth of a percent** — device medians 3231.7 →
2996.6, 5995.3 → 5582.5 and 11715.4 → 10932.0, which is −7.3, −6.9 and −6.7%
again. **The null pair was run first**, same word both arms, seven pairs:
**+0.6% on the wall and 0.0% on the device**, ranges across, no claim on either.

### What it is against the ceiling

The instruction issues where it issued — the four register arms are within a
tenth of the sitting T2 and T3 read them on:

    what issues                            device            rate
    the matrix instruction, four held     16.55ms    25.3 TFLOP/s
    the same chain held once               4.73ms    22.2 TFLOP/s
    the same four on 16-bit operands      15.86ms    26.4 TFLOP/s
    scalar fused multiply-adds             4.55ms    23.0 TFLOP/s

    a prefill of                production   rounded    of the instruction
     2048  q_proj, tiled            3.76ms    3.46ms      72.1% → 78.4%
     2048  a routed bank           16.62ms   15.44ms      49.0% → 52.7%
     4096  q_proj, tiled            7.50ms    6.92ms      72.3% → 78.4%
     4096  a routed bank           26.15ms   24.46ms      62.2% → 66.5%
     8192  q_proj, tiled           14.98ms   13.81ms      72.4% → 78.6%
     8192  a routed bank           52.10ms   48.80ms      62.5% → 66.7%

**824.6 GFLOP at 52.10 ms against the same at 48.80** — 15.83 TFLOP/s against
16.90, which is **62.5% of the instruction against 66.7%.** T3 stopped at 62.4%
and T2 at 61.9%; the two levers those milestones were sent for moved it half a
point between them and this moves it four.

**What it does not do is move the ceiling.** The instruction's own 16-bit row is
26.4 TFLOP/s against 25.3 — four percent — so what this buys is not issue rate.
It is the threadgroup traffic a lane's own read pulls, which is the term the
per-lane read left standing and the one T2's hoisting arm priced at 6%.

### What the new entries cost to compile

**No entry was added, which is the first thing to say and is a fact rather than a
measurement.** `rounded` compiles the same two production entries a model load
builds, from a source that differs by a typedef and a stride:

    a block of                                            entries  compiling  the source
    32 codes a step, the elements a lane owns, float            2    1955 us    56.0 KiB
    64 codes a step, the elements a lane owns, float            2    1993 us    56.0 KiB
    32 codes a step, through simdgroup_load, float              2    1906 us    55.5 KiB
    32 codes a step, the elements a lane owns, half             2    1964 us    56.0 KiB

**The operand word is 9 µs over the shipped block on a two-millisecond compile,
out of a source the same size to a tenth of a kilobyte.** An earlier sitting read
it 30 µs *under*; both are the same answer, which is that a typedef is not a
build cost. D1 found that merely reordering which entries are created first is
worth 2.4% of a 2048-token prefill, so this is the row that had to be checked,
and what it says is that the word costs nothing to build. The clock is a warm
compile and says so — every row is timed behind one shipped build.

What this milestone did add to the *shipped* block's own compile is 42 µs and
0.9 KiB, against T3's 1913 µs and 55.1 KiB: the typedef and the three casts. That
is 0.0008% of a 5.6-second prefill, paid once at load.

## Prefill's ordering, which the encoder already had

**D4's mechanism is in the encoder and the encoder is not gated on a shape.**
`Batch` opens every pass `MTLDispatchTypeConcurrent` and puts a barrier in front
of any dispatch touching something the open group wrote, deriving the answer from
each kernel's own Metal source. Nothing about that asks what is being run. **So a
prefill has had it since the day it landed, and what was missing was the
reading.**

T2's limiter table puts barriers at 10% of the prefill matmul and a prefill
issues over a thousand dispatches with far more independent work than a one-row
step, where 546 of 670 groups were singletons. **That expectation is wrong, and
the width distribution is what says so rather than a mean:**

    a prefill of  dispatches  passes  groups  barriers  a group  widest  of them
        97               996      21     668       647     1.49       4     65%
      2048               993      43     665       622     1.49       4     63%
      8192               993      43     665       622     1.49       4     63%

    504 of 668 groups are one dispatch at 97, 501 of 665 at 2048 and at 8192,
    and the rest divide 42x2, 80x3, 42x4

    a decode step at  dispatches  passes  groups  barriers  a group  widest  of them
        97 / 2048 / 8192       918      14     670       656     1.37       4     71%

    546 of 670 groups are one dispatch, and the rest divide 42x2, 40x3, 42x4

**They are the same sequence with different shapes in it.** A prefill and a
decode step both walk 42 layers of the same dependency graph — a norm feeds four
projections, a convolution feeds a norm, an activation feeds a `down` — so a
prefill's extra 75 dispatches buy it 40 more three-wide groups and nothing else.
**75% of a prefill's groups are one dispatch against a decode step's 81%**, and
63% of its dispatches still sit behind a barrier.

**And what it is worth there is 1.4%, decaying to nothing.** D4's own two refs,
paired under `--numerics production`, seven alternating pairs:

    a prefill of   serial pass    concurrent, barriered    change
     2048            3286.3 ms          3240.3 ms         -1.4%   7 of 7 apart
     8192           11798.9 ms         11762.2 ms         -0.3%   ranges across

Against a decode step, where D4 measured the same lever at **16.204 → 15.200 ms,
−6.2%**. The two readings have one explanation and it is the shape of the
dispatches rather than the shape of the graph: a decode step's dispatches are
microseconds and leave the machine idle, so overlapping two of them hides a
launch; a prefill's are milliseconds and already fill it, so overlapping two of
them hides nothing. **The concurrency a prefill has is the same concurrency, and
there is less to buy with it.**

**So this closes.** The mechanism generalises, it was already generalised, it is
worth about a percent at the opening a coding session has and nothing at a long
one, and a prefill's groups are as narrow as a decode step's. The
derived-versus-encoded barrier count is now held at prefill shape as well as at
decode shape, out of one helper — a missing barrier is a race that is correct
most of the time, and that count is the only thing that catches it.

### What did not move

**No token changed and no default changed.** The flag stays defaulted to
`reference`, the reference path is byte-identical — nothing behind the flag is
compiled under it — and the recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what all three words and both
backends write.

**`production` is where T3 left it, to the digit.** Paired against the commit
this milestone opened at, both arms under `--numerics production`, seven pairs at
2048 tokens: device **3232.2 ms against 3231.9**, ranges across, no claim; the
wall reads +1.2% with the ranges across and 5 of 7, which is this instrument's
own 0.6% null and not a change. **A typedef that reads `float` is a typedef that
compiles to what was there before**, and the source-level case is what says it
rather than the clock.

**A decode step is untouched, which it has to be**: the block is not dispatched
at any shape a decode step has. Seven pairs at the recorded prompt's own context:
**16.315 ms against 16.325**, device **15.224 against 15.223**, ranges across, no
claim. Five pairs at a 2048-key context: device **20.082 against 20.029**, ranges
across, no claim.

**And the tiers pass at the ends of this series**: 692 hermetic, 692 gated, 74
timing, nothing failed. One of the 74 did fail on the first run of it and is the
error this milestone's own review missed — three ablation arms in
`what_a_prefills_blocked_matmul_is_bound_by` match the block's source by hand and
two of them matched lines the operand width rewrote. `instead_of` refused them
rather than reporting the shipped kernel under another name, which is what it is
for. The limiter table it prints is where T2 left it: the barriers at 91 and 92%,
the hoisted loads at 100%, the weight at 97 and 98%.

**Speculation is where it was.** `bench sweep` on the packed heads, one sitting:

    k    a token   tokens/s   speedup   accepted
    0    16.271ms     61.5      1.000
    1    15.622ms     64.0      1.042   84.85%
    2    15.383ms     65.0      1.058   86.96 78.26%
    3    15.207ms     65.8      1.070   85.00 65.00 55.00%
    4    18.346ms     54.5      0.887   82.35 64.71 52.94 47.06%

**`k = 3` is still the best depth and every acceptance figure is the one this
file records**, at 1.829, 2.560, 2.909 and 3.368 tokens a round.

**A session is where K1 left it, at both openings**, three pairs each, ours under
`--numerics production` with both engines keeping their cache:

    opening        ours kept   mlx-vlm kept   behind
    2048            18.03 s       14.35 s     1.26×
    8192            31.83 s       25.44 s     1.25×

    the per-turn bookkeeping inside it
    2048             1.045 ms     31.228 ms
    8192             1.098 ms     64.023 ms

**K1's kept cache is flat in the context where theirs is linear** — about a
millisecond a turn against 31 and 64.

**Against the reference, one sitting**, on the packed heads under
`--numerics production`, and both engines wrote the same first eight tokens at
every one of the four prompts:

    prompt × answer   ours p50   mlx-vlm p50   ahead on p50
        97 × 128       16.37ms       23.06ms          1.41×
       385 × 128       17.51ms       23.32ms          1.33×
       769 × 128       18.43ms       23.74ms          1.29×
        97 × 512       17.50ms       23.49ms          1.34×

Speculating two deep, the same sitting: **14.45 ms a token at 97 × 128 and 14.36
at 97 × 512**, against the reference's 23.06 and 23.49 — 1.60× and 1.64×.

**Our own column there is the standalone reading and the other engine's is the
paired one, and that is a fact about the instrument this milestone found.** Under
`just bench-engines`'s alternation the two *long-prompt* rows read **24.40 and
24.89 ms** where a run of the same binary alone reads 17.51 and 18.43 — while the
two 97-token rows read 16.43 and 17.11 against 16.37 and 17.50, which is nothing.
So it is not this engine getting slower: it is the other engine's residency
between two of our arms, and it reaches only the runs whose prompt is long enough
to touch a lot of the checkpoint. **What says it is not this milestone** is a
paired decode step at a 2048-key context across the commit this opened at: device
**20.082 ms against 20.029**, ranges across, no claim, five pairs. The
cross-engine rows above are quoted standalone and labelled; `bench-engines`'s
alternation is what a *ratio* wants and it is contaminating an absolute.

**And what the operand word would be worth to a session, which is the shape a
user feels**, three pairs at the 2048 opening, both arms keeping:
**18.167 s under `production` against 17.560 under `rounded`, −3.3%, 3 of 3 with
the ranges apart** — the whole of it in the opening prefill, which is 7.67 s
against 7.40. That would put the 2048 opening at about 1.22× behind rather than
1.26×.

### The floor this stops at, and what is under it

**The instruction issues at 25.3 TFLOP/s and that is where it issued.** The
routed bank at 8192 runs 824.6 GFLOP: **52.10 ms and 15.83 TFLOP/s under
`production`, 48.80 and 16.90 under `rounded` — 62.5% of the instruction against
66.7%.**

What that leaves under the 33%, re-read from T2's terms: the barriers are 10%,
reachable by evening out or hiding what a threadgroup stages rather than by
having fewer of them, and a double buffer is the candidate that leaves open; the
fragment loads are 6%, reachable only by holding a fragment across steps, which
T2 measured at 101% on both shapes; the three byte-count terms are 8%. **The
operand width was the largest priced thing left on this kernel and it is now
measured and spent.**

**What is not left is prefill's ordering**, which is measured at 1.4% and already
taken, and **what is not left is a decode step**, which D5 closed at 71% of its
own bandwidth floor.

**What this does not license is a claim that the engine is finished.** It is
1.25× behind at the 8192 opening and 1.26× at 2048 under `production`, and about
1.22× at 2048 under a word nobody has adopted. The gap is real and the next thing
after it is continuous batching, which nothing here starts.

### What this milestone leaves the owner

**One decision, with both halves of it measured.**

- **What it costs**: a product that used to be exact carries eleven bits of
  significand instead of twenty-four; the drift is 1.5 to 2.1e-4, flat in the
  reduction, two decades above the block's own; and the sentence this file's
  numerics section is written on stops being true of one of its three words.
  There is no oracle on that side and there cannot be one.
- **What it buys**: 6.7 to 7.3% of a prefill's device time, 4.6 to 5.9% of its
  wall, 3.3% of a coding session's opening turn, and four points of the
  instruction ceiling.
- **What it does not cost**: any token, over 384 sampled argmaxes across six
  prompts and six distributions; the recorded continuation; a decode step; a
  compiled entry; 30 µs of compile.

**Nothing here makes it a default and nothing here recommends one.** What the
flag's third word is *for* is that the question can now be asked with numbers on
both sides of it.

## The two levers between 62% and 74%

**Both were priced off the instruction mix rather than off a build, both are
built and measured here, and together they are worth 0.8% where they were priced
at 16%.** T2 left the routed bank at 62% of an instruction ceiling nothing can
beat, named the two terms worth taking next — a 64-code step at the barrier arm's
10%, mlx-style per-lane fragment loads at the hoisting arm's 6% — and put the two
at their ceilings at 74%. **61.9% is where this milestone opened and 62.4% is
where it stops.**

**One of the two is refused and one of the two ships**, and they fail in the same
shape: each was priced off an ablation arm, and each arm prices the *memory
traffic and the waiting behind it* rather than the instruction that expresses it.
A lever that changes only the expression reaches neither.

**And the one that ships re-opened a door T2 closed, which is the largest thing
here.** 16-bit operands were 5 to 9% behind the block through `simdgroup_load` and
are 6 to 8% *ahead* of it per lane, because that read is the one where the operand
width reaches the bytes a lane pulls. It is not taken — the drift it was refused
on has not moved and the exact-product property is what this flag is written on —
but it is on the table at a size neither lever reached.

### What a step of 64 codes is worth, which is less than nothing

A step is `MMA_CODES_A_STEP` codes wide and carries two barriers, so a reduction
of 4096 is 128 steps and **256 barriers a pass**. A step of two quantisation
groups halves that count, and the barrier arm — the one that removes them and
races — prices the whole term at 10%, so 5% was the ceiling.

It is a value the block carries now, beside the height, the width and the
threadgroup T1 made values. Warm, swept both ways, every arm answering the
shipped kernel bit for bit except the two that race:

    a prefill of 8192 tokens
    a step of                        declares      q_proj, tiled      a routed bank
    32 codes                         13.5 KiB    15.08ms   100%     52.52ms   100%
    64 codes                         25.5 KiB    16.52ms   110%     56.03ms   107%
    32 codes, declaring the wider    25.5 KiB    15.93ms   106%     55.15ms   105%
    32 codes, racing                 13.5 KiB    13.42ms    89%     47.48ms    90%
    64 codes, racing                 25.5 KiB    14.92ms    99%     51.29ms    98%

**The third row is what makes this a diagnosis rather than a refusal.** It is the
shipped step declaring the wider one's threadgroup memory and never touching the
extra — `RESIDENCY`'s own trick pointed the other way, since what a threadgroup
declares decides how many of them a core holds. It carries the wider step's
occupancy and none of its benefit, and it costs **5%**. So the staging is what a
wider step is bought at, and the arithmetic of the trade is on the table.

**And the last two rows say the barrier term is not the barriers' count.** At 32
codes the barriers cost 52.52 ms against 47.48 racing, which is 5.04 ms; at 64
codes, with *half as many of them*, they cost 56.03 against 51.29, which is 4.74.
**Halving the count returned 0.30 ms — six percent of the term and 0.6% of the
call — where the staging it needed cost 2.63 ms.** What a barrier waits is the
spread between the first simdgroup to arrive and the last, and a step twice as
wide puts twice as much staging behind each one: the count halves, the imbalance
behind each doubles, and the wait is conserved.

The same reading at 2048 and 4096, to a percent. Two groups is also as far as
this can be asked at all — the staging is `(rows + cols)` rows of the step and its
padding, so 32 codes declares 13.5 KiB, 64 declares 25.5 and 128 would declare
49.5, past the 32 KiB a threadgroup gets.

**So the lever is refused, and with it every lever whose mechanism is a smaller
count.** The 10% the arm prices is the spread a threadgroup's staging leaves, so
what could still reach it is an arrangement that evens that spread or hides it —
a double buffer, where one tile fills while the other is multiplied, is the
obvious candidate and is *not* refused here, because what it changes is what a
barrier waits on rather than how many there are. What is refused is the reading
that made a wider step look worth half the term.

### What reading a fragment per lane is worth, which is a twelfth of its price

`gather_qmm`'s `BaseMMAFrag::load` never calls `simdgroup_load`. It reads the two
elements each lane owns straight out of threadgroup memory through a static
lane-to-coordinate map, and takes the transpose by swapping the two strides it
indexes with — the one structural difference between mlx's inner loop and this
one, and the reason the hoisting arm's 6% was thought reachable.

**The map is measured against this part rather than inherited.** Three lines of a
header are a claim about which two elements of an 8×8 fragment a lane holds, and
this is hardware nobody here has a document for — **a map naming somebody else's
two elements would not crash and would not read zero**, it would read two staged
floats, multiply them, and answer a plausible number of the right magnitude. So
one tile goes through both forms in one kernel and the two are held equal lane for
lane and element for element, plain and transposed, at both widths the 16-bit arm
puts these same two calls through. The tile's own coverage is checked beside that
agreement, because the two forms share the tile if not the map: a fill striding by
a width the probe was not dispatched at had both of them agreeing about the same
wrong thing, and the 64 distinct values against a padding none of them takes is
what caught it.

Warm, swept both ways, bit for bit against each other:

    a fragment read                     q_proj, tiled      a routed bank
    through simdgroup_load            15.08ms   101%     52.54ms   101%
    the two elements a lane owns      14.96ms   100%     52.18ms   100%

**0.7 to 0.9% at every shape and every length, and it ships.** In the model,
paired and alternating, seven pairs, under the production flag — the two ends at
the commit before this one and 4096 across the whole milestone, which is the same
claim because the step commit's own pair read **11751.0 ms against 11755.9 at
8192 with the ranges across**:

    prefill device        before      after   change   ranges
     2048               3240.7ms   3231.7ms    -0.3%   apart, 7 of 7
     4096               6021.3ms   5995.9ms    -0.4%   apart, 7 of 7
     8192              11760.6ms  11713.9ms    -0.4%   apart, 7 of 7

**A twelfth of what it was priced at, and the arm that priced it is what says
why.** Hoisting leaves three of every four fragments unread and reads 6%, so what
it prices is three of every four *reads of threadgroup memory*. The per-lane form
issues the same reads of the same bytes and changes only the instruction that
carries them, so 0.8% is the instruction and 5.2% is the traffic — which only a
fragment held across steps could reach, and T2 measured a block twice as tall at
101% on both shapes at 8192.

### The door the per-lane read re-opened, which is worth more than either lever

**16-bit operands were refused twice and one of the two reasons has gone.** A8
refused them on principle — this flag's two sides sum the same exact products in
different orders, and a 16-bit operand rounds the product itself — and T2 refused
them again on the clock, at 5 to 9% behind the block at equal occupancy. The
clock half was measured against a block that read its fragments through
`simdgroup_load`, and this milestone changed that.

The same arm under both reads, each against the block compiled with the same
read, warm and swept both ways:

    a fragment read through simdgroup_load
    at                              the block   16-bit at 36     at 40     at 44
    2048   q_proj, tiled               3.79ms            +9%       +9%       +5%
    2048   a routed bank, grouped     16.80ms            +9%       +9%       +5%
    4096   q_proj, tiled               7.56ms            +9%       +9%       +5%
    4096   a routed bank, grouped     26.36ms            +8%       +8%       +4%
    8192   q_proj, tiled              15.09ms            +9%       +9%       +5%
    8192   a routed bank, grouped     52.54ms            +8%       +8%       +5%

    a fragment read per lane, which is the shipped one
    at                              the block   16-bit at 36     at 40     at 44
    2048   q_proj, tiled               3.77ms            -2%       -2%       -8%
    2048   a routed bank, grouped     16.67ms            +0%       +0%       -7%
    4096   q_proj, tiled               7.50ms            -1%       -1%       -8%
    4096   a routed bank, grouped     26.12ms            -0%       -1%       -6%
    8192   q_proj, tiled              14.98ms            -1%       -1%       -8%
    8192   a routed bank, grouped     52.17ms            -1%       -1%       -6%

**The operand width reaches the traffic under one read and not under the other.**
A lane pulling the two elements it owns pulls four bytes at half where it pulls
eight at float; a `simdgroup_load` is one instruction over the whole fragment
either way, so under it the width reached the registers and nothing else. That is
the same finding as the per-lane lever's own, read from the other side: what these
two levers move between them is threadgroup-memory traffic, and only the one that
moves *bytes* is worth anything.

**So the refusal now rests on the drift alone, and the drift has not moved.**
1.5 to 2.1e-4 across reductions of 32 to 4096, flat — which is the operand
arriving already rounded and not a chain getting longer — against the block's own
1.3e-7 to 1.6e-6, which grows with the reduction because a fragment accumulator is
one running sum. Two decades.

**This is not taken here and the reason is scope rather than arithmetic.** 6 to
8% at the padding that wins is larger than both levers this milestone was sent for
put together, and it is the largest priced thing left on this kernel. What it
costs is the sentence this file's numerics section is written on — that both sides
form the same exact products and differ only in the order they are summed. A
16-bit operand ends that, so it is a third numerics rather than a faster
production path, and A8's own words for it still hold: it would have to move into
the conditioning table rather than sit beside it. **What has changed is that it is
now a 6 to 8% question rather than a −5% one**, and that is a different question
to decline.

### What the two of them are worth against the ceiling

The instruction issues where it issued — the four register arms land within a
hundredth of a millisecond of the sitting T2 read them on, which is what says the
denominator did not move under the numerator:

    what issues                            device            rate
    the matrix instruction, four held     16.55ms    25.3 TFLOP/s
    the same chain held once               4.75ms    22.1 TFLOP/s
    the same four on 16-bit operands      15.86ms    26.4 TFLOP/s
    scalar fused multiply-adds             4.55ms    23.0 TFLOP/s

And against it, one sitting each side of the milestone:

    a prefill of                    before             after      of the instruction
     2048  q_proj, tiled            3.79ms            3.76ms         72% → 72%
     2048  a routed bank           16.79ms           16.65ms         48% → 49%
     4096  q_proj, tiled            7.55ms            7.50ms         72% → 72%
     4096  a routed bank           26.41ms           26.16ms         62% → 62%
     8192  q_proj, tiled           15.09ms           14.98ms         72% → 72%
     8192  a routed bank           52.56ms           52.13ms         62% → 62%

The column prints 62% on both sides of the milestone; the milliseconds behind it
are 824.6 GFLOP at 52.56 ms against the same at 52.13, which is **61.9% against
62.4%**. **74% did not arrive**: of the 16% the two levers were priced at, one
*cost* 6.7% and is refused and the other returned 0.8%.

### What the compile cost

**No entry was added, which is the first thing to say and the one that is a fact
rather than a measurement.** D1 found that merely reordering which of this
source's entries are created first is worth 2.4% of a 2048-token prefill, so a
lever that made the entries longer to build would spend its winnings before a call
reached them. Every arm of both sweeps builds the two production entries a model
load builds, and the case asserts the count:

    a block of                                  entries   compiling   the source
    32 codes a step, the elements a lane owns         2     1913 us     55.1 KiB
    64 codes a step, the elements a lane owns         2     1911 us     55.1 KiB
    32 codes a step, through simdgroup_load           2     1878 us     54.6 KiB

**35 microseconds and half a kilobyte**, against a prefill that is seconds. The
clock is a warm compile and says so: the first build in a process carries the
compiler's start-up, so every row is timed behind one shipped build.

### The per-kernel table, and against the reference in one sitting

Under `--numerics production`, one sampled prefill a row:

    tokens   the grouped matmul   the row-tiled matmul   both   of every pass
     2048           1.81s                1.20s          3.01s        88%
     4096           3.15s                2.37s          5.52s        85%
     8192           5.87s                4.67s         10.54s        82%
    16384          11.30s                9.26s         20.56s        80%

**The share is where T2 read it and the seconds are two to four percent under**,
which is this milestone and a sitting between them and not separable at that
size — the paired instrument above is what claims, and it claims 0.3 to 0.4%.

`just bench-engines` on the packed heads, three pairs, both engines alternating,
ranges apart at every row, and both engines wrote the same first eight tokens at
every one of the four prompts:

    prompt × answer   ours p50   mlx-vlm p50   ahead on p50
        97 × 128       16.42ms       22.87ms          1.39×
       385 × 128       17.69ms       23.13ms          1.31×
       769 × 128       18.54ms       23.54ms          1.27×
        97 × 512       17.55ms       23.09ms          1.31×

Speculating three deep, the same sitting: **13.77 ms a token at 97 × 128 and 13.06
at 97 × 512**, against the reference's 22.98 and 23.16 — 1.67× and 1.77×.

**And a session, both openings, both engines, one sitting each**, three pairs,
ours under the production flag with both engines keeping their cache:

    opening        ours kept   mlx-vlm kept   behind
    2048            18.11 s       14.28 s     1.27×
    8192            31.52 s       25.49 s     1.24×

    the per-turn bookkeeping inside it
    2048             0.947 ms     30.588 ms
    8192             1.022 ms     68.954 ms

**K1's kept cache is where K1 left it** — ours flat in the context where theirs is
linear, at about a millisecond a turn against 31 and 69.

### What did not move

**No token changed and no default changed.** The flag stays defaulted to
reference, the reference path is byte-identical — neither lever is compiled under
it — and the recorded continuation `[656, 13, 623, 180069, 86333, 60500, 220, 23]`
is what both backends write. `just diverge` puts the corpus through both numerics
and reads **384 of 384 tokens agreed, no prompt parted**, which is the gate the
production path owes before any timing claim.

**A decode step is untouched, which it has to be**: the block is not dispatched at
any shape a decode step has. Paired against the commit this milestone opened at,
seven pairs: **16.295 ms against 16.299**, device **15.216 against 15.213**, ranges
across, no claim. **The null pair was run first and read 0.1% on the wall and 0.0%
on the device**, ranges across — same binary both arms, seven pairs.

**Speculation is where it was, digit for digit.** `bench sweep` on the packed
heads, one sitting:

    k    a token   tokens/s   speedup   accepted
    0    16.388ms     61.0      1.000
    1    15.654ms     63.9      1.047   84.85%
    2    15.485ms     64.6      1.058   86.96 78.26%
    3    15.430ms     64.8      1.062   85.00 65.00 55.00%
    4    18.350ms     54.5      0.893   82.35 64.71 52.94 47.06%

**`k = 3` is still the best depth** and every acceptance figure is the one this
file records, at 1.829, 2.560, 2.909 and 3.368 tokens a round. The absolute
column sits about 6% above the rows quoted elsewhere, which is this sitting rather
than this milestone — the paired decode above is what says the step did not move,
and it read no claim.

**Our own prefill wall, unsampled, under the production flag**, one sitting:

    tokens    wall      device
     2048    5.60 s     3.23 s
     4096    9.05 s     6.00 s
     8192   16.73 s    11.71 s

### The floor this stops at, and its arithmetic

**A stopping condition is a floor you can prove, and T2's is still the one this
stops at — the two levers it named are now measured rather than priced.**

The instruction issues at 25.34 TFLOP/s on this part and that is the same rate its
scalar fused multiply-add issues at, so no arrangement of this arithmetic goes
faster. The routed bank at 8192 runs 824.6 GFLOP: **52.56 ms and 15.69 TFLOP/s
before, 52.13 and 15.82 after — 61.9% of the instruction against 62.4%.** The
column prints 62% on both sides, which is the honest way to read it.

**74% was the arithmetic and it was arithmetic about the wrong thing.** T2 put the
two levers at 16% between them by reading two ablation arms — the barrier arm at
10% and the hoisting arm at 6% — and both readings are correct about what they
measure. What is wrong is the step from an arm to a lever:

- The barrier arm removes the barriers and races, so it prices **the waiting**. A
  wider step halves the count and doubles what each one waits behind, and the term
  went 5.04 ms to 4.74 — **6% of it, where halving the count predicted 50%.**
- The hoisting arm leaves three of every four fragments unread, so it prices
  **three of every four reads of threadgroup memory**. A per-lane read issues the
  same reads and rewrites the instruction around them, and returned 0.8% of a 6%
  term — **a twelfth, where the arm's own share predicted all of it.**

**So the rule this milestone leaves is about the instrument rather than about the
kernel: an ablation that removes work prices the work, and it bounds only a lever
that also removes the work.** A lever that keeps the work and rewrites the
instruction around it collects the instruction alone, and on this part the
instruction is the smaller part of both terms priced here.

What that leaves under the 38%, re-read: the barriers are 10%, reachable by
evening out or hiding what a threadgroup stages rather than by having fewer of
them, and a double buffer is the candidate that leaves open; the fragment loads
are 6%, reachable only by holding a fragment across steps, which is the taller
block T2 measured at 101% on both shapes at 8192; the three byte-count terms are
8%; and about a fifth is unattributed. **Removing every priced term entirely still
puts the call at 82% and mlx's quantised kernel is at 79%**, so the distance is
where T2 left it and the two named levers across it are closed with measurements
rather than open with prices.

**And the largest priced thing left on this kernel is now the operand width**, at
6 to 8% under the shipped read, behind a drift two decades above the block's own
and behind the exact-product property this flag is documented on. That is a
question about what `--numerics production` is allowed to mean rather than about
the matmul, and it is not this milestone's to answer.

**What this does not license is a claim that the engine is finished.** It is
1.24× behind at the 8192 opening and 1.27× at 2048, and the gap is real. What it
says is that the two levers a reader of T2 would have taken next are worth 0.8%
between them, that the third thing is where the unattributed fifth is, and that
the next person to price a lever off an ablation arm should ask first whether the
lever removes the work the arm removed.

### What this milestone leaves

**The block now carries seven values and the last of them is not a shape.** The
height, the width, the threadgroup and its two simdgroup extents are T1's; the
step is this milestone's and is still a shape; the fragment read is the first axis
here that changes the instruction the kernel issues rather than what one
threadgroup covers. Every one of the seven is swept by a case that holds its arms
bit for bit against the shipped block, and `distinct` is what keeps an arm from
quietly becoming the shipped one when a sweep's subject moves.

**What that is for is the thing after this**, which is continuous batching: a
batch adds a dimension to every shape-keyed decision this engine makes, and these
are where the matmul's now live. Nothing here starts it.

## The walk a decode step could stop doing serially, and why it cannot

**Splitting an output element's walk between simdgroups is the one decode lever
D1 named and would not take, it is built and measured here, and it does not
land.** A decode step under it is **21.080 ms against 16.289**, device **19.949
against 15.203** — seven alternating pairs of one build under the two words,
every pair the same way, ranges apart, against a null pair that read +0.1% and
0.0%. The lever is 31% the wrong way, and the reason is not the summation order
it was held back for. It is that this kernel's whole speed is that a threadgroup
of it shares nothing, and a cross-simdgroup reduction has to share something.

**Twice, on two builds and at two cut bounds.** The figure above is the best the
lever was made to read, at four cuts. The first build allowed eight, and seven
pairs of it against the tree that removed it — two refs, both arms under
`--numerics production` — read **23.494 ms against 16.258** and **22.337 against
15.141** on the device, 7 of 7 with the ranges apart. Bounding the cut at four
recovered a third of the loss and none of the lever.

**Nothing ships.** The two entries are compiled out of a source only the sweep
builds, which is where D3 left the indirect command buffer for the same reason:
mechanism sound, arithmetic against it, did not build. The reference path and the
production path are both where they were, to the bit.

### Why it looked like the lever, and it was not a bad reading

An output element is one simdgroup here, and
`whether_a_decode_dispatch_is_short_of_the_machine` puts the rate on the element
count and nothing else: 512 elements reach 12% of the bus and 131072 reach 95%.
Every shape a decode step has is on the steep part of that curve — its widest
bank is 24576 elements and its narrowest projection 512 — and the head is the
only call it makes that is not.

So the arithmetic was: `n` simdgroups on one element are `n` times the simdgroups
in flight over the same bytes, each holding `1/n` of the row and issuing its own
loads, and what the walk waits on is the loads it has outstanding rather than the
bandwidth behind them — which
`what_a_decode_steps_packed_matmul_is_bound_by` measured at a fifth to two fifths
a term. **That reading still stands. What it left out is what holding a partial
costs.**

### What a cut costs, which is not the cut

`what_a_packed_multiply_costs_at_each_count_its_walk_is_cut_into` is the sweep,
warm and both ways, against the entry the cut stands in front of. **The column
that decides it is the first one**: a cut of *one*, which partitions nothing —
the same lanes, the same width, the same chunks, and
`the_cuts_of_a_split_walk_read_each_byte_of_the_row_exactly_once` holds that the
bytes are the same bytes.

    shape             elements     uncut     1 cut    2 cuts    3 cuts    4 cuts   1 cut, no barrier
    r_proj                 512     3.7µs      108%      118%      139%      139%       110%
    k_proj, v_proj        1024     5.3µs      118%      119%      146%      152%       126%
    q_proj, o_proj        4096    14.9µs      127%      137%      168%      174%       139%
    routed gate, up      12288    41.3µs      128%      138%      169%      177%       138%
    lm_head             200058   638.6µs      130%      140%      175%      182%       140%

**A cut of one is already 8 to 30% and it has cut nothing.** So the sweep divides
into two questions, and both have answers.

**The barrier is not it.** The last column is the same source with
`threadgroup_barrier` taken out — which races, and is here to be priced rather
than run — and taking it out gives none of the cut of one back: 110 to 142%
against that column's 108 to 130%, which is no better and on four of the five
shapes a little worse. **Read across two sittings rather than down one column**,
because that is what the arm's own spread needs: the first sitting had the two at
115-143% and 116-142% and the second at 108-130% and 110-142%, so the pair moves
together and the gap between them is inside what either moves by.

**The threadgroup width is not it either.** A cut of one puts a threadgroup at 32
threads where the plain entry takes 64, and
`what_a_packed_multiply_costs_at_each_threadgroup_an_untiled_call_takes` has had
those two as one arm since D1: 3.6 against 3.6 µs, 5.3 against 5.1, 14.5 against
14.2, 39.0 against 38.9.

**What is left is the threadgroup array.** Thirty-two bytes of it, written once
and read once by the same thread at a cut of one, costs 27 to 30% at the shapes
with elements enough to be near the bus. **That is a term this file has priced
before on the other kernel and never on this one**: A8's residency sweep found a
threadgroup staging *two keys* of a tile costing what one staging nineteen costs,
and wrote down that what the left arm of that curve punishes is the staging
rather than the amount of it. This is the same shape of answer at 32 bytes, which
is as small as the amount gets.

**And it is exactly what `THREADS_AN_ELEMENTS_GROUP` has said about this entry
since D1**, in a sentence written to explain why its threadgroup is only a
scheduling knob: *a threadgroup of `packed_matmul` is that many independent
simdgroups and nothing else — no threadgroup memory, no barrier, no shared load,
no fragment anything shares.* That is the property being spent, and there is no
version of a cross-simdgroup reduction that keeps it.

**And the cut on top of that is real too**: four cuts are another 31 to 52 points
over one, which is the walk falling under `CHUNKS_IN_FLIGHT`. A lane holds
four chunks before it adds any, the held loop needs four strides to fit under the
row, and the checkpoint's 4096-code row is sixteen steps — so past four cuts each
lane is back to issuing four bytes and waiting a memory round trip for them,
which is exactly the wait D1 removed.

### The other way to hold a partial, priced and not built

Attention folds its splits with a second dispatch, and that version needs no
threadgroup memory at all: each cut writes its partial to device memory and a
fold reads them. **The arithmetic is against it before it is built.** A decode
step makes **377** of these calls — 297 untiled and 80 paired, which is what D2's
and D3's merges left of D1's 457 — and a fold apiece is 2.60 µs of device time.
That is **0.98 ms added to a 15.20 ms step, 6.4%, before the fold does any
work**, and per call it is 70% of the narrowest one it would fold and 6% of a
routed bank's. D2, D3 and D4 spent three milestones removing 204 dispatches to
buy 1.00 ms between them; this would put 377 back to buy something the sweep
above says is negative.

### What the cut is worth to the answer, which is the column it does not lose

The block this flag puts in front of the matmul is the worse-conditioned of the
two orders, by a factor that grows with the reduction: a fragment accumulator
carries the whole thing as one running sum, and at 4096 codes it drifts 4.1e-6
against the reference's 1.4e-7. **A cut is the other direction** — `n` partials,
each a lane-strided walk of `1/n` of the row, added at the end — and it is the
first alternative accumulation of this walk that stays inside the *reference's*
own bound rather than needing `MMA_TOLERANCE`. Against an f64 accumulation of the
same products:

    reduction   the reference   four cuts
         1024      7.8e-8         9.2e-8
         4096      1.3e-7         9.2e-8

**Which of the two is nearer is a wash at the short reduction and the cut at the
checkpoint's own**, and the reason is in the shapes: at 1024 codes each cut holds
one chunk and the tail that adds the four partials is as long as the walk that
fed it. `a_cut_walk_is_f32_noise_at_every_reduction` therefore asserts
`TOLERANCE` and reports the pair rather than asserting an ordering between them.

**It is the column nobody was asking about**, and it is recorded because a lever
refused on speed should still say what it was worth on everything else — and
because it is the one thing here that would matter if the cost above ever moved.

### The second candidate, which was a misreading and is measured anyway

**A4's limiter table has been read as putting the band at 28% of a decode step
against 9 to 23% at a prefill.** Its four cells are 2048 and 8192 *tokens* on a
global and on a windowed layer — two kinds of layer at one regime — and 28% is
the windowed column at both lengths where 23% and 9% are the global one. A decode
step appears nowhere in it. What separates those columns is the band's own
extent: a windowed layer's live keys are all inside the 1024-distance band where
a global layer at 8192 reads mostly keys further back and takes the early return.

So the question was open rather than answered, and
`what_a_decode_steps_attention_is_bound_by` is A4's instrument at one query row
over the keys a sequence holds, which is the shape a step has.

    without                     385 global   385 window   8192 global   8192 window
    nothing — the kernel          104.75µs     106.88µs      812.79µs      265.33µs
    the keys and values           78.38µs 75%  77.62µs 73%  499.25µs 61%  225.58µs 85%
    the band it derives           71.12µs 68%  70.75µs 66%  640.79µs 79%  256.25µs 97%
    the exp's precision           84.92µs 81%  84.46µs 79%  708.00µs 87%  283.38µs 107%
    the exp                       85.75µs 82%  86.38µs 81%  686.62µs 84%  157.12µs 59%
    the barriers                  45.75µs 44%  45.92µs 43%  424.25µs 52%  161.25µs 61%
    the tile's two reductions     43.83µs 42%  42.62µs 40%  385.67µs 47%  147.58µs 56%
    the weighting                 49.50µs 47%  43.38µs 41%  305.83µs 38%  133.88µs 50%
    three quarters of the dot     50.92µs 49%  47.88µs 45%  402.96µs 50%  158.50µs 60%
    the simd_sum                  55.29µs 53%  57.62µs 54%  523.50µs 64%  188.12µs 71%

**Two rows did not reproduce and are printed rather than dropped**, which is the
discipline A4's own `simd_sum` row is under: a second sitting reads the barriers
at 78% where this one has 44%, and the exp at 108% where this one has 59%. The
band's four cells repeat to a point across both sittings, and so do the tile's
two reductions, the weighting and the dot.

**The band at a decode shape is 32 to 34% at 385 keys, 21% at 8192 global and 3%
at 8192 windowed** — so it is bigger than the prefill's windowed 28% at short
context and collapses at long context on the layers that are 35 of 42. Against
the in-model share `which_kernels_own_a_decode_step_at_each_context` gives it —
windowed attention 9.1% of a step at 385 keys and global 2.0% — removing the band
*entirely* is an upper bound of **3.7% of a decode step**, and removing it
entirely answers wrongly, because it is the mask.

**It is not the term with most to gain and it is not close.** The tile's two
reductions are 44 to 60% of a decode-shaped call at every cell, ahead of the band
at all four, and that is a trade this file made deliberately at 32 entries and
against a barrier. **What the decode column actually says is that no term here is
free** — the cross-lane reduction, which A4 measured at 100% of a prefill and
called nothing, is 29 to 47% here, and the transcendental is 16 to 19% on the
three cells that reproduced where A4 read 101%. A decode step's attention is 32
threadgroups on an 80-core part; it is waiting on itself, and every instruction
taken out of it shows.

### What the instrument gained, which is the thing that would have caught this

**Every entry behind this flag had been gated on a height until this milestone
tried to put one on the decode path**, and that is what made a prompt-length
floor a sufficient guard: `bench diverge` held each of the corpus's six prompts
against `SHORTEST_BLOCKED_CALL` before it ran anything, and a corpus that cleared
it reached everything the flag selects.

**An entry a decode step dispatches is gated on nothing.** A call of one row
reaches it, so no prompt length says whether a differential run ever ran it — and
a run that never did would report its 384 agreeing argmaxes exactly as loudly as
one that ran it at every step. That is the same "a thing agreeing with itself"
the floor exists to refuse, arriving on the axis the floor does not watch.

So each kernel now names what it put behind the flag, and a differential run
holds what it *dispatched* against those lists, over the two regimes a generation
has — a prefill and a decode step — per prompt, because only the corpus's longest
member reaches the grouped entry. `Encoded` gains the compiled entry beside the
label it is charged to, because a layer's attention step is `windowed attention`
whichever kernel carried it and the label cannot answer which one ran.

It reports, and this run reports:

    the flag selected mma_attention, mma_matmul_grouped, mma_matmul_rows
    at a prefill and nothing at a decode step

**Which is the true sentence about this tree**, and was the true sentence about
every tree before it — stated by the instrument rather than inferred from the
predicates. While the cut was on the path it read `split_matmul,
split_matmul_pair` at a decode step, which is how the run that measured the
divergence knew it had measured anything.

### What did not move

**No token moved.** `just diverge` over the corpus: six prompts, 64 tokens each,
**384 of 384 agreed**, nothing parted. That is the same figure A7 recorded and it
is the same figure for the same reason — with the cut off the path the flag
reaches no decode dispatch, and the reference and production paths run the same
untiled kernel.

**The reference path compiles the string it compiled before**, which
`the_reference_source_does_not_carry_the_production_entries` holds against the
list the flag itself keeps rather than against a second copy of it — so an entry
put behind the flag and left out of that list is a case that fails rather than a
check that quietly shrank. `a_call_under_either_numerics_answers_what_the_other_
answers` asserts an untiled call is the same *bits* under both words again, which
is what the cut's removal restores and what its presence correctly broke.

**A decode step is where D4 left it**, on a host settled to 0.32% on the wall and
0.25% on the device before the sitting. `what_a_decode_step_costs_as_the_context
_grows`, one sitting, beside D4's own row:

    context   D4's step   here   median   D4's device   here   tokens/s   peak
        97     16.80ms  16.50ms  16.47ms     15.37ms  15.11ms     60.6   0.28 GiB
       385     18.40ms  17.87ms  17.79ms     16.99ms  16.32ms     56.0   1.31 GiB
       769     18.81ms  18.65ms  18.65ms     17.42ms  17.10ms     53.6   1.32 GiB
      2048     21.23ms  21.40ms  21.46ms     19.81ms  20.07ms     46.7   2.05 GiB
      4096     22.77ms  23.08ms  22.93ms     21.38ms  21.62ms     43.3   2.81 GiB
      8192     25.69ms  25.47ms  25.44ms     24.26ms  24.05ms     39.3   4.30 GiB

**The mean and the median agree to fifteen hundredths of a millisecond at every
row**, which is what says no row has a step in it that is not a decode step. The
385 row reads 17.87 against the four sittings this file records at 17.95 to
18.77, and it is the row this file has already said not to read absolutely.

**And both words agree at a long context**, which is what the flag reaching no
decode dispatch looks like from the outside: seven pairs at 8192 keys read the
device at **24.241 ms against 24.174**, ranges across, no claim. The wall row on
that measurement reads 47.396 against 48.270 and also claims nothing — it is the
row `bench decode --context` charges a prefill's deferred step to, and the device
column is the one to read there.

**Speculation is where it was, on the checkpoint whose row it is.** `bench sweep`
on the packed heads, one sitting:

    k    a token   tokens/s   speedup   accepted
    0    16.061ms     62.3      1.000
    1    14.847ms     67.4      1.082   84.85%
    2    14.704ms     68.0      1.092   86.96 78.26%
    3    14.459ms     69.2      1.111   85.00 65.00 55.00%
    4    17.785ms     56.2      0.903   82.35 64.71 52.94 47.06%

**`k = 3` is still the best depth** and the acceptance rows are digit for digit
the ones this file records, at 1.829, 2.560, 2.909 and 3.368 tokens a round. The
default checkpoint's bfloat16 heads were run too and reproduce *their* row —
84.8%, 91-74%, 84-74-63%, 82-65-53-47% — which is the two-depths-apart signature
this file uses to catch a swapped checkpoint, landing where it should.

**Prefill is unmoved**, one sitting under the production flag against the three
lengths this file quotes: **5.561 s** wall and 3.239 s device at 2048 against the
recorded 5.74 and 3.39, **9.092** and 6.021 at 4096 against 9.33 and 6.26, and
**16.827** and 11.772 at 8192 against 17.29 and 12.19 — every one of them a few
percent under, uniformly, which is a sitting rather than a change.

**And K1's kept cache is where K1 left it**, which is the one architectural
advantage this engine has measured. `just bench-session`, three pairs, cold
against kept, every pair the same way and the ranges apart:

                        K1's figure     here    range
    a cold session         265.53 s   263.739  263.672-263.783
    a kept session          71.36 s    71.162   70.666- 71.440
    a kept turn's marks     1.135ms     1.124    1.095-  1.177

**Every row reproduces to under a percent**, which is what a milestone that
touched nothing between two requests should read. What the pair itself says is
−73.0% on the session, and −36.7% on the per-turn bookkeeping against the cold
arm's own 1.775 ms — that second figure is cold against kept and not either
against K1, which is the comparison the third row above makes.

**The tiers are 681, 681 and 70**, against the 672, 672 and 68 this milestone
opened on: seven cases hold the differential run's new reach check, two hold the
cut's partition and its drift, and the two measurements are the cut's sweep and
the decode-shaped limiter table. `cargo nextest run` against a real checkpoint is
681 of 681 in 819 s with nothing skipped but the measurements, and the recorded
continuation, D4's derived barrier count, both block floors and
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` are all inside
it.

### The floor this stops at, which is the one D1 left

**The matmul is where D1 left it: 5.91 GB of packed weight read once a token,
8.15 ms of it at 725 GB/s, and 71% of that reached.** Nothing here moved it in
either direction, because nothing here shipped. The sampled per-kernel table puts
the matmul's two entries at 377 calls, 5932 MB and 67.5% of a step's device time
— which is the *share* D1 quotes, and is read as a share rather than as an
absolute for the reason that table states about itself.

**What the cut would have done to that column is the finding rather than a
footnote.** Its rate falls at every shape it touches — 302 to 218 GB/s on
`r_proj`, 421 to 276 on `k_proj`, 598 to 343 on `q_proj`, 648 to 367 on a routed
bank — so the lever aimed at the gap between 67% and 100% takes the kernel to
about 55% of where it was. D1's arithmetic that a matmul at the bus leaves a step
at 13.5 ms device still stands, and this is one fewer way to get there.

**What is left under it is what D1 said and one thing more.** D1 named the
remaining 3.4 ms above floor as per-dispatch fixed cost and unhidden walk, and
took the dispatch half over three milestones. The walk half now has a measured
obstacle rather than an untried lever: **the walk cannot be shortened by sharing
it, because sharing costs more than the walk does.** What is not ruled out is
shortening it by giving one simdgroup *more than one output element* — a tile
across columns, where the rows a decode step has are one and the columns are
thousands, and where what is shared is the input read rather than a partial sum.
`what_a_decode_steps_packed_matmul_is_bound_by` puts the input's loads at 21 to
43%, and that is the term such a tile would reach. **Nothing here measures it**,
and it is written down because this milestone is what makes it the next question
rather than the second one.

### What this milestone leaves

**Two refusals and an instrument.** The cut is refused on a measurement rather
than on an argument, and the band is refused on a measurement that also corrects
what the argument was. What is left behind is the differential run knowing which
entries it actually exercised, which is the thing that would have caught a decode
entry behind the flag that no corpus ever ran — and which now prints, on every
run, the sentence that says so.

**The decode-shaped limiter table is the thing to read next.** No term in it is
free where A4's prefill table had two that were nothing, and the largest is a
reduction this engine took on purpose at 32 entries. That is a share and not a
lever, which is exactly the distance A4's own 29% was from a kernel — and the
milestone after prefill's ceiling work is still continuous batching.

## The ordering under the dispatches

**A decode step's 918 dispatches were ordered end to end, and 656 of the 904 gaps
between them needed to be.** Metal's pass is `MTLDispatchTypeConcurrent` now,
with a `memoryBarrierWithScope:` in front of any dispatch that touches something
the dispatches since the last barrier wrote: **17.254 ms to 16.287**, device
**16.204 to 15.200**, seven alternating pairs, every pair the same way and the
ranges apart, against a null pair that read 0.0% and −0.1%. That is **−5.6% on
the wall and −6.2% on the device**, and the wall follows the device again for
D3's reason — the device is 92% of a step, so it is the only side worth being on.

### Where the barrier goes, which is derived and not decided

A barrier removed by inspection is a race that is correct most of the time. This
engine's kernels carry state between calls — a convolution's window, the
attention span, the router's selection — so a missing one answers the step it was
made on correctly and a later step wrongly, which is a rejection away rather than
a token away. So nothing in `crate::ordering` asks what a dispatch is *for*. It
asks two things:

- **which allocations filled its slots**, which the command buffer's own bindings
  say; and
- **which of those slots the kernel may write**, which the kernel's own Metal
  source says — `constant T &x` is an address space that cannot be written,
  `device const T *x` is a pointer that cannot, `device T *out` is one that may,
  and `device T *const out` is one that may as well, since `const` counts only
  where it qualifies the pointee. `Kernel::writes` parses that at compile time
  off the string the pipeline was built from, and the Metal compiler is what
  enforces it. A parameter it does not recognise comes back written, which is a
  barrier kept; an entry it cannot find at all is a panic, because the silent
  failure would be a step that reads as fully ordered.

Two dispatches may share a group when no allocation they both name is written by
either — a read after a write, a write after a read and two writes are all
hazards, and all three close a group. The identity is the binding's address,
which is sound for as long as it has to be: **a command buffer retains everything
bound into it**, so nothing named in an open pass can be freed under it. Both
ways that identity can be wrong are written down where it is used, and the only
one that could *drop* a barrier needs two `MTLBuffer`s over one writable region —
which this engine never makes, since the one wrap it does is a checkpoint's
read-only pages.

**The same `Open` answers for the analysis and for the encoder.** The division
`crate::ordering` reports offline and the barriers `Batch` encodes as it goes are
one rule asked at two times rather than two that can drift, and
`what_a_decode_steps_dispatches_have_to_wait_for` asserts the two counts are
equal on a real step at all three contexts. A count nothing checked is a claim.

### What the division comes to

Identical at 97, 2048 and 8192 keys:

    918 dispatches, 14 passes, 670 groups, 656 barriers, 1.37 to a group

Every group wider than one is nameable, which is what says the division found the
model rather than an artefact of the analysis:

    42 of packed_matmul x4                   a layer's q, k, v and r
    40 of packed_matmul, packed_matmul,      both banks' down, beside the softmax
          router_weights                     over the eight logits the top-k named
    40 of router_top_k, packed_matmul_pair   the selection, beside the shared
                                             bank's pair, which does not need it
     2 of packed_matmul, packed_matmul       a dense layer's gate and up

and **546 of the 670 groups are one dispatch**: a decode step at one row is a
chain, and what concurrency there is to find is 124 places wide.

### Why the arithmetic said this would lose

**The lever was priced before it was built and the price had the wrong sign.**
`what_a_barrier_costs_the_device_against_the_dispatches_it_separates` is a
thousand empty dispatches through one pass on the driver's own clock, swept by
how many of them share a group:

    a thousand empty dispatches                barriers        each   disagreed
    a serial pass                                     0     1.554µs          1%
    a concurrent pass, a barrier each              1000     2.401µs         27%
    a concurrent pass, 2 to a group                 500     1.216µs          3%
    a concurrent pass, 3 to a group                 333     0.809µs          1%
    a concurrent pass, 4 to a group                 250     0.621µs          2%
    a concurrent pass, 8 to a group                 125     0.384µs          1%
    a concurrent pass, no barriers                    0     0.339µs          0%

**An explicit barrier is dearer than the ordering it replaces** — 2.06
microseconds against 1.22 — so only a barrier *removed* is worth anything, and a
group has to average 1.70 dispatches before a concurrent pass overtakes a serial
one. A decode step's groups average **1.37**. That arithmetic says a step would
get **0.24 ms slower**, and D3's own closing figure — the barrier at 91% of what
an empty dispatch costs — says the same thing from the other mechanism.

It is 1.00 ms faster. **What a serial pass costs a dispatch that computes
something is not its launch, it is the execution it keeps from overlapping**, and
a probe over dispatches that compute nothing cannot see that term because there
is nothing to overlap. Three arms on the real step, each its own sitting of seven
alternating pairs against the serial pass, every pair the same way and the ranges
apart:

    what the pass was                       serial   this arm   change   tokens
    concurrent, a barrier after every one  16.121ms  16.825ms    +4.4%   right
    concurrent, the derived 656 barriers   16.204ms  15.200ms    -6.2%   right
    concurrent, no barriers at all         16.101ms   9.252ms   -42.5%   wrong

Each row carries its own serial arm because each is a paired sitting of its own,
and the three agree on it to half a percent.

**The bottom row is the ceiling and it is not a milestone**: it races, and it
writes `!!! ruhig.Statement!! presa` where the oracle writes `'s daughter is
the`. What it prices is the ordering — **6.85 ms of a 16.10 ms device step,
against the 1.12 ms the fixed cost implies** — and the factor of six between them
is the term the empty-dispatch sweep is blind to.

**The second row is the control, and it costs nothing to be sure of**: a
concurrent pass with a barrier after every dispatch is the same ordering the
serial type gives, so it writes the same tokens, and what it measures is the
mechanism alone. 0.70 ms over 904 barriers is **0.78 microseconds a barrier**, so
the 656 that stay hand about 0.51 ms back — which puts the overlap the 124 wide
groups recover at about **1.51 ms of the 6.85 available, 1.00 of which survives
the mechanism.**

**The finer barrier buys nothing.** `memoryBarrierWithResources:count:` naming
only the allocations the open group touched, rather than every buffer the pass
may have written, reads **−6.1% against −6.0%** in two sittings taken back to
back, and 2.457 µs against 2.401 on the empty sweep. It is the sharper instrument, the
hardware does not make the distinction, and the coarse one ships: it hands Metal
no pointer back and it orders strictly more.

### What a decode step costs now, at every context

The paired instrument, three pairs at each of six contexts, device only for the
reason `bench decode --context` carries — it charges the step a prefill deferred:

    context   device before   device after   change   pairs
         97        16.235ms       15.242ms    -6.1%   3 of 3, ranges apart
        385        17.510ms       16.618ms    -5.1%   3 of 3, ranges apart
        769        18.382ms       17.408ms    -5.3%   3 of 3, ranges apart
       2048        20.939ms       19.940ms    -4.8%   3 of 3, ranges apart
       4096        22.293ms       21.312ms    -4.4%   3 of 3, ranges apart
       8192        25.020ms       24.102ms    -3.7%   3 of 3, ranges apart

**A fixed 0.89 to 1.00 ms at every length**, which is what removing a count of
orderings rather than a rate has to look like: the same absolute saving, shrinking
as a fraction only because attention grows underneath it.

`what_a_decode_step_costs_as_the_context_grows`, one sitting, against D3's:

    context     before      after   tokens/s   of floor   device before   device after
       97      17.60ms    16.80ms  56.8→59.5    46%→49%        16.35ms        15.37ms
      385      18.86ms    18.40ms  53.0→54.3    43%→44%        17.57ms        16.99ms
      769      19.79ms    18.81ms  50.5→53.2    41%→43%        18.47ms        17.42ms
     2048      22.68ms    21.23ms  44.1→47.1    36%→38%        21.20ms        19.81ms
     4096      23.90ms    22.77ms  41.8→43.9    34%→36%        22.54ms        21.38ms
     8192      26.54ms    25.69ms  37.7→38.9    31%→32%        25.18ms        24.26ms

The two sides are two sittings and the claim is the paired table above it. The
mean and the median agree to fifteen hundredths of a millisecond at every row,
and to two hundredths on five of the six — 18.40 against 18.55 is the widest and
25.69 against 25.68 the closest — which is what says no row has a step in it that
is not a decode step.

**The 385 row is the noisy one and this file should say so**: four sittings of it
read 17.95, 18.01, 18.40 and 18.77 ms where every other row repeats to within a
tenth. The paired instrument above, which is the claim, reads it at 16.618 ms of
device against 17.510 and does not share the spread.

### What speculation costs now

Five alternating pairs on the packed heads:

    depth    a token before   a token after   tokens/s   device before   device after
    k = 0         17.186ms        16.274ms   58.2→61.4        16.137ms        15.174ms
    k = 1         15.474ms        14.919ms   64.6→67.0        14.263ms        13.765ms
    k = 2         15.312ms        14.763ms   65.3→67.7        13.880ms        13.434ms
    k = 3         16.296ms        14.645ms   61.4→68.3        14.536ms        13.036ms
    k = 4         18.726ms        17.317ms   53.4→57.7        16.627ms        15.517ms

**The depth that pays best moves from two to three.** Against that run's own
`k = 0` the speedups are 1.091, 1.102, **1.112** and 0.940, where they were
1.111, 1.122, 1.055 and 0.918 — so `k = 3` gains where the shallower depths give
a little back, and the reason is the same one the whole section is about: a
deeper round puts more rows through each dispatch, so `k = 0` is the arm with the
most idle machine to fill and it is the arm that improves most.

**Acceptance is identical in both arms at every depth** — 84.85% at one; 86.96
and 78.26 at two; 85, 65 and 55 at three; 82.35, 64.71, 52.94 and 47.06 at four —
and so is what a round banks. Which is what a scheduling change has to read as:
the heads guess what they guessed and the model verifies what it verified.

### Against the reference, one sitting

Three pairs, both engines alternating, the packed heads:

    prompt x generated    ours p50   the reference p50   ahead
    97 x 128              16.42ms              22.90ms   1.39x
    385 x 128             17.68ms              23.12ms   1.31x
    769 x 128             18.53ms              23.52ms   1.27x
    97 x 512              17.52ms              23.08ms   1.32x

Ahead of D3's 1.32, 1.25, 1.21 and 1.25 at every row. Speculating two deep, the
same sitting: **14.67 ms a token at 97 × 128 and 14.39 at 97 × 512 against the
reference's 23.01 and 23.16**, which is **1.57× and 1.61×** where D3 read 1.53
and 1.57. Both engines wrote the same first eight tokens at every one of the four
prompts.

### What did not move

**No token changed, on any run.** The recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both backends write,
`--backend cpu` included, and the numerics flag stays defaulted to reference.

**And it is the same continuation through a rewind at every depth, over and over.**
This is the first change here that can produce a race rather than a wrong number,
and bit-identity on one run does not prove it. So
`speculating_writes_the_text_that_decoding_one_token_at_a_time_writes` was driven
**ten times** — five on an idle host and five with another process holding the
GPU — which is fifty generations across `k` of 0, 1, 2, 3 and 4, every one of
them writing the oracle's own text. Every depth rejects guesses on the way
through: 2 of 3 accepted at one, 2 of 6 at two, 2 of 8 at three, 2 of 10 at four.
Beside that, **116 further generations** driven straight through the binary at
`k` of 1 to 4 over three prompts, under one and then two concurrent MLX matmul
loops, each held against its own `k = 0` oracle: all 116 matched.

**Prefill did not move.** Three pairs each:

    tokens     wall    device   ranges
     2048     +0.2%     -0.1%   across on the wall, no claim
     4096     -0.0%     -0.0%   across, no claim
     8192     +0.2%     -0.1%   across, no claim

Which is what a decode-step lever should read as, and for a sharper reason than
D2's count: a prefill's dispatches are hundreds of rows each, so they already
fill the machine and there is no overlap for a concurrent pass to recover.

**K1's kept cache is untouched**: a simulated coding session reads **265.53 s
cold against 71.36 s kept** — every turn prefilling the same 321 tokens where the
cold arm re-prefills 8832 to 9472 — against D3's 265.23 and 71.20, and the
per-turn bookkeeping is **1.135 ms**, inside the 1.013 to 1.267 this file has at
the two ends of a context sweep.

**Both MMA floors,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert`,
`the_bounded_loop_is_the_unbounded_one_bit_for_bit` and
`a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine` are where
they were**, and so is
`a_paired_dispatch_answers_what_the_two_dispatches_it_replaces_answer`.

### The floor this stops at, and what is under it

**A decode step still reads 5.91 GB of packed weight once, and at 725 GB/s that
is 8.15 ms.** Taken at 97 keys, where the dispatch count and the barrier count
are both known, the step is at **53% of it on the device where it was at 50%**:

    term                                        cost   what it is
    the weights, at the bus                   8.15ms   the floor
    918 launches, at 2.60µs each              2.39ms   being 918 dispatches
    the 656 barriers, at 0.78µs each          0.51ms   the ordering that is real
    everything else                           4.19ms   the walk, and the bytes above the floor
    the step                                 15.24ms

The second row is D3's figure and is not re-measured here: 2.60 µs was taken by
removing dispatches from a *serial* pass, and what a launch costs in a concurrent
one is a different measurement that nothing in this milestone turns on.

**What is under the third row is not another 0.51 ms.** The ceiling arm says the
ordering is worth 6.85 ms; this step's own paired saving of 0.99 ms plus the 0.51
its barriers hand back recovers **1.50** of it; the rest is held by 546
groups of one dispatch, and those are a real dependency chain rather than a count
that has not been merged yet. Taking more of it would mean changing what a decode
step computes rather than how it is submitted — which is what the next milestone
is: **continuous batching**, the original differentiator, still unbuilt, and the
only thing left that changes the engine's category rather than its constant
factor. A second sequence in flight is exactly the thing that fills the 546.

## The encode a decode step could stop doing, and what it would be worth

**A decode step encodes the same thousand dispatches every step, that sequence
is reusable, and patching one is an order of magnitude cheaper than encoding it.
The milestone still does not exist, for two reasons that are each on their own
sufficient: the device is already executing for 92% of a step, and an indirect
command costs the device twice what the dispatch it replaces costs.**

This was opened to remove the 7.79 ms of host-side encoding D2 measured in an
18.22 ms step, on the reading that a step freed of it would approach the 10.43 ms
it spends inside a wait — **roughly 1.75×, larger than every lever D1 and D2
shipped combined.** The reading is wrong, and it is wrong in a way that retires a
whole class of levers rather than only this one.

### The ceiling on every host-side lever, which is 8%

`where_a_decode_step_spends_its_time`, one run, one step:

    a 17.70ms decode step, 1000 dispatches in 14 submissions over 997 buffers
    operation           calls   self time   share
    submit and wait        28     10.33ms   58.3%
    dispatch encode       999      6.14ms   34.7%
    unaccounted                    1.22ms    6.9%
    of which the device reported executing for 16.29ms, 92.0% of the step

**10.43 ms was never the device's floor.** It is what this process *blocked*
for, and a run that commits at a layer boundary and keeps encoding blocks only
for the part of the GPU's work it has run out of other things to overlap. The
device's own clock says it executed for **16.29 ms of a 17.70 ms step**, summed
over fourteen submissions of one serial queue — so 16.29 ms is a floor the step
cannot go under whatever this side does.

**That leaves 1.41 ms.** Every host-side lever this project could reach — a
reused command sequence, buffers that are not reallocated every step, a cheaper
shape struct — divides up 1.41 ms of a 17.70 ms step. **8.0% is the ceiling on
all of them together**, and it is the same arithmetic whichever is tried.

It also explains D2's own result rather than leaving it a puzzle. Eighty fewer
dispatches took 0.33 ms off the device and about 0.3 ms off the encode, and the
wall moved 0.06: the encode was never on the critical path, so the half of that
change which landed on the host bought nothing at all.

### What patching a command costs against encoding one fresh

Asked anyway, because the mechanism is what the next milestone to want it will
have to know. `what_patching_an_indirect_command_costs_against_encoding_it_fresh`,
a thousand of each, warm, every wait taken outside the clock, median and mean:

    what                                              each   a thousand    mean
    a dispatch encoded into a pass, and committed   1.554µs      1.554ms  1.521ms
    the same dispatch through this crate's own path 1.562µs      1.562ms  1.573ms
    an indirect command written whole               1.395µs      1.395ms  1.406ms
    every one of its five slots rebound             0.775µs      0.775ms  0.782ms
    one slot of it rebound                          0.184µs      0.184ms  0.186ms
    a command reached for rather than held          0.145µs      0.145ms  0.146ms
    a pass executing one of them, and committed          —      0.007ms  0.007ms
    a pass executing the thousand, and committed         —      0.008ms  0.011ms

**Patching is an order cheaper than encoding: 0.184 µs against 1.554.** And the
handover is a fixed cost rather than a per-command one — one command executed
costs 0.007 ms and a thousand cost 0.008 — so nothing on this side walks the
sequence. Both of the last two rows are the whole pass: open a command buffer,
declare, execute, commit.

**A command reached for costs what patching it costs**, which is the one trap in
the API: `indirectComputeCommandAtIndex:` builds an Objective-C object every time
it is asked, so the commands have to be taken once and held or the patch pays
twice.

**This crate's own path is what the raw calls are**, 1.562 µs against 1.554 —
so none of a step's encode is bookkeeping this crate could have removed without
Metal.

The saxpy kernel this is measured through binds five buffers where this engine's
kernels bind six to a dozen, so the fresh-encode row is *cheaper* than the
dispatches it stands for and every ratio here is the conservative one.

### What it would cost the device, which is where it dies

`what_the_device_makes_of_a_thousand_indirect_commands`, the same thousand empty
dispatches three ways, on **the driver's own clock** — `GPUEndTime` less
`GPUStartTime`, because every arm ends in a wait and a wall time around one
carries the commit and the wake, which at these durations is the same order as
what is being measured:

    what                                            each   a thousand    mean
    a thousand dispatches through one pass       1.093µs      1.093ms  1.218ms
    a thousand indirect commands, a barrier each 2.210µs      2.210ms  2.211ms
    a thousand indirect commands, no barriers    0.205µs      0.205ms  0.205ms

**An indirect command that means what the dispatch it replaces means costs the
device 2.02× as much.** A pass's dispatches are serial and an indirect buffer's
are concurrent, so a sequence that preserves the ordering asks for a barrier on
every command — and that arm is 2.210 µs against the encoded 1.093.

A decode step is 1042 dispatches, so a reused sequence would add about **1.2 ms
to the 16.29 ms the device already spends**, against saving at most 1.41 ms on a
host that is not the constraint. **It would make a decode step slower**, and no
amount of cheaper patching changes that: the two figures are on opposite sides of
the only clock that binds.

The 0.205 µs row is the same mechanism with the ordering taken off, and it is a
different finding — see the end of this section.

### The two things it would cost that are affordable, measured anyway

`what_a_pipeline_built_for_an_indirect_command_costs`:
`supportIndirectCommandBuffers` is set on a descriptor before a pipeline is built
and cannot be told to one afterwards, so an engine on this path compiles every
kernel carrying it. It is free — **1.001× up the list and 1.000× down it** over a
50 MB saxpy at 371 GB/s, and the same 1024-thread threadgroup either way.

`what_declaring_a_steps_allocations_to_an_indirect_pass_costs`: a command inside
an indirect buffer does not tell the driver what it binds, so the pass executing
it has to name every allocation any command may reach. That is a cost in a step's
*distinct buffers* rather than its dispatches, and it is nearly free at the scale
this engine has — **0.010 ms for 256 and 0.102 ms for 4096**, and a residency set
would remove it entirely.

### Whether the sequence is reusable, which is yes at every context

`what_changes_between_two_decode_steps` writes down every dispatch a step
encodes — the entry, the pipeline, the grid, and what fills each argument slot —
and holds consecutive steps against each other. The worst pair at each context:

    context  dispatches  commands     slots    bound   inline  metal encode  the row
         97        1042         0      6327     2688      168        2.32ms   6.23ms
       2048        1042         0      6327     2864      168        2.28ms   6.11ms
       8192        1042         0      6327     2458      168        2.37ms   6.24ms

**`commands` is zero at every context and every pair.** Not one dispatch changes
its entry, its pipeline or its grid between two steps, and the count does not
move — so the concern about attention's span is answered: a decode step's
attention takes its span as a value a kernel reads rather than as a grid, and
8192 keys dispatch what 97 keys dispatch. The numerics flag and `splits_for` pick
their kernels at a shape a decode step always has, and freeze nothing that was
right for another. **So the premise of the lever holds and the lever still does
not pay**, which is worth separating: what fails is the arithmetic, not the
structure, and a later milestone that changes the arithmetic inherits a sequence
already known to be reusable.

**What varies is 2458 to 2864 of 6327 slots, and it varies because the engine
allocates rather than because the model does.** 997 buffers a step is what the
step table counts, and a slot naming a fresh allocation is most of that column;
the 168 inline slots are the shapes and the spans, which move by a token a step.
Those counts are floors — a slot is compared by the address it named, and this
engine frees and reallocates enough that an address can come back — and a floor
is the direction a decision *against* reusing the sequence can be made under.

**The last two columns are the other half.** Of the 6.14 ms `dispatch encode`
row, the Metal calls are **2.32 ms**; the remaining 3.8 ms is this side — 997
allocations, the shape structs, the routing — and an indirect command buffer does
not touch any of it. A reused sequence would replace 2.32 ms with about 2700
`setKernelBuffer:` calls, which at 0.184 µs is **0.50 ms**, plus somewhere to put
168 inline arguments, which an indirect command has no binding for at all.

### What is kept

**The probe, and the instrument.** `crate::indirect` prices the mechanism and
asserts that a command written into an indirect buffer answers what the dispatch
it stands for answers, exactly; `crate::trace` is what any later milestone that
wants to reuse a sequence has to run first, and it is off unless somebody asks.
Neither is on any path the model takes.

**What is not kept is the plan**: every kernel recompiled behind a pipeline flag,
168 inline arguments given allocations they do not have, a sequence replayed
after a speculation rewind — for a step that would get slower.

### The one thing this found that is worth more than what it was looking for

The third row of the device table is the same mechanism with the ordering taken
off: **2.210 µs a command with a barrier, 0.205 µs without.** D2 named Metal's
serial dispatch type, put "most of the 4.4 microseconds" on it, and left it; this
is that estimate measured from inside a mechanism that lets the barrier be
addressed one command at a time, and it says the ordering is **91% of what an
indirect command costs the device when it computes nothing.**

A decode step pays 4.2 ms of its 16.29 for being a thousand dispatches, and the
device's clock is the one thing here that does reach the wall. It stays left, for
the reason D2 left it: it puts a wrong answer one missing barrier away with
nothing in the arithmetic to catch it. What has changed is that it is no longer
an estimate, and that **it does not need an indirect command buffer** —
`computeCommandEncoderWithDispatchType:` offers the same concurrency to the
encoder this engine already has, without the 2.02× an indirect command costs.

**That last sentence is what the next milestone took up, and it held** — see "The
ordering under the dispatches", where the ordering comes off and the barriers
that remain are derived from what each dispatch reads. The 4.2 ms above is what a
step pays for being a thousand dispatches at all; the ordering alone turns out to
be **6.85 ms**, larger than the whole of it, for the reason that section gives: an
empty dispatch pays the barrier's launch and a real one pays the overlap it
loses.

## The three pairs a decode step had left

**A decode step made 1000 dispatches and it makes 876**, and this time the wall
followed the device: **17.301 ms to 16.978**, device **16.292 to 15.970**, seven
alternating pairs, every pair the same way and the ranges apart, against a null
pair that read −0.1% on both and a host settled to 0.8% on the wall before the
sitting. A step's submissions go from 14 to 11 with it.

The 124 are the three pairs D2 priced and left. Each is two calls that read
different rows against different weights into different places, neither reading
anything the other writes:

    what                                      merges   dispatches   the entry
    a layer's query norm and key norm             42           42   rms_norm_pair
    a layer's key and value convolutions          42           42   short_conv_pair
    a MoE layer's routed and shared SwiGLU        40           40   swiglu_pair

**All three keep the bits by construction rather than within a tolerance**, and
each has a case holding the merged dispatch against the two it replaces exactly.
The branch is on the thread's or the threadgroup's own position, so a thread runs
the walk it ran alone over the same values in the same order; what a merge
changes is which of two shapes it reads.

### What a dispatch's fixed cost actually is, which is 2.6 µs

D2 put it at 4.15 microseconds, from 80 dispatches removed for 0.332 ms of
device time. **124 removed here take 0.322 ms, which is 2.60 µs each** — and the
difference is not noise, it is that the two are not the same shape of merge.

D2's pair doubled the output elements of the dispatch that survived — a routed
bank's 12288 to 24576 — and every shape a decode step has is on the steep part of
that kernel's rate curve, so its 4.15 was a launch *and* a width. **None of the
three here wins any width**: a norm's threadgroups, a convolution's threads and
an activation's threads are the same threads over the same elements whichever
dispatch they are in. 2.60 µs is the launch alone, and it is the number a
count-based estimate should be built on.

It also prices what is left. **876 dispatches at 2.60 µs is 2.28 ms of a 15.97 ms
device step** — still the largest single thing a decode step pays for that no
kernel computes.

### Why the wall moved, where D2's did not

D2 took 0.33 ms off the device and the wall moved 0.06. This takes 0.32 off and
**the wall moves 0.32** — the same saving, transmitted where the other was not.

What separates them is which side of the step each landed on. The encode is 31.9%
of a step against a device executing for 92.5% of it, so the wall follows the
device and only the device. D2's fusion moved both — 0.33 ms of device and about
0.3 of encode — and the encode half bought nothing, because a host that finishes
early only waits longer. **A device saving is worth its whole self here and a
host saving is worth almost nothing**, which is the same arithmetic the section
above draws the 8% ceiling from.

The step table either side, one sitting each, says the same thing from the other
end:

    a decode step               dispatches   submit and wait   dispatch encode   device
    before                            1000    10.33ms  58.3%    6.14ms  34.7%   16.29ms
    after                              876    11.04ms  62.0%    5.67ms  31.9%   16.47ms

**The encode row falls by 0.47 ms and the wait rises by 0.71**, which is a host
that ran out of things to do sooner rather than a step that got slower — the two
sittings are unpaired and the claim is the paired figure above.

### What each merge is, and the one that is not a kernel

**The query norm and the key norm** read `[queries, heads, head_dim]` and
`[queries, kv_heads, head_dim]` against two weights into two landings — one a
buffer the attention step reads, one the span the layer keeps. What they have to
share is the threadgroup, which the width decides, and both are `head_dim` here.
A checkpoint that made them differ would want two threadgroup shapes out of one
grid, which Metal cannot dispatch — so `norm::encode_pair` makes the two
dispatches it always made, and that arm is a case rather than a hope.

**The key and value convolutions** need nothing to agree. That kernel gives a
thread to a channel of a timestep, declares no threadgroup memory and takes no
barrier, so there is no threadgroup shape for two calls to disagree about. What
its case has to hold instead is the *window*: a convolution carries two and
alternates between them, and one that answered the right rows while swapping the
wrong window would pass on the first call and be wrong on every one after it.
Three consecutive calls of one sequence, each way, is what pins it.

**The two banks' SwiGLU is the only one of the three that is a change to the
layer.** A bank's activation sits between its pair and its `down`, so a layer
that finished one bank before starting the other had a `down` between the two
activations and nothing to merge. Both banks' pairs are dispatched first now,
then one activation over both, then both downs — the same order every read
already required, and the reason this one is in `experts.rs` rather than in a
kernel.

The per-kernel table shows all three arriving, at a step's own shapes:

    kernel              calls      device   share   what it replaced
    rms_norm_pair          42    520.68µs    2.8%   84 of the 169 rms_norm calls
    short_conv_pair        42    383.24µs    2.0%   84 of the 126 short_conv calls
    swiglu_pair            40    220.04µs    1.2%   80 of the 82 swiglu calls

### What a decode step costs now, at every context

`what_a_decode_step_costs_as_the_context_grows`, one sitting, against D2's:

    context     before      after   tokens/s   of floor   device before   device after
       97      18.06ms    17.60ms  55.4→56.8    45%→46%        16.68ms        16.35ms
      385      19.37ms    18.86ms  51.6→53.0    42%→43%        18.01ms        17.57ms
      769      20.04ms    19.79ms  49.9→50.5    41%→41%        18.69ms        18.47ms
     2048      22.82ms    22.68ms  43.8→44.1    36%→36%        21.45ms        21.20ms
     4096      24.03ms    23.90ms  41.6→41.8    34%→34%        22.61ms        22.54ms
     8192      26.86ms    26.54ms  37.2→37.7    30%→31%        25.47ms        25.18ms

The mean and the median agree to six hundredths of a millisecond at every row —
17.60 against 17.64, 26.54 against 26.59 — which is what says no row has a step
in it that is not a decode step.

**The two sides are two sittings and the claim is the paired instrument**, which
was taken at each of the same six contexts, three pairs each, and reads the
device down every one of them:

    context   device before   device after   change   pairs
         97        16.597ms       16.275ms    -1.9%   3 of 3, ranges apart
        385        18.066ms       17.677ms    -2.1%   3 of 3, ranges apart
        769        18.696ms       18.385ms    -1.7%   3 of 3, ranges apart
       2048        21.419ms       21.000ms    -2.0%   3 of 3, ranges apart
       4096        22.752ms       22.408ms    -1.5%   3 of 3, ranges apart
       8192        25.559ms       25.250ms    -1.2%   3 of 3, ranges apart

A fixed 0.31 to 0.42 ms at every length, which is what removing a count rather
than a rate has to look like. The wall column of that instrument is not quoted,
for the reason `bench decode --context` carries: it charges the step a prefill
deferred, which at 8192 is 47 ms against a true 26.5.

### What speculation costs now

Five alternating pairs on the packed heads:

    depth    a token before   a token after   tokens/s   device before   device after
    k = 0         17.568ms        17.188ms   56.9→58.2        16.510ms       16.151ms
    k = 1         15.816ms        15.465ms   63.2→64.7        14.468ms       14.251ms
    k = 2         15.502ms        15.291ms   64.5→65.4        13.995ms       13.865ms
    k = 3         16.409ms        16.135ms   60.9→62.0        14.639ms       14.474ms
    k = 4         18.599ms        18.745ms   53.8→53.3        16.670ms       16.645ms

**The device row is a claim at `k` of 0 through 3 and no claim at 4**, and the
wall is a claim at 0 and 1. The depth that pays best is still two.

`k = 4` is where the sitting runs out of resolution rather than where the merge
stops applying: its wall ranges are 17.98–19.18 against 18.43–19.16 and two of
five pairs go each way. Nothing about the three merges depends on the depth — a
round of `k + 1` rows dispatches the same 876 kernels over taller calls, so what
falls is the *share* a launch is of each, from 2.0% of a step at `k = 0` to
0.15% at four.

**Acceptance did not move and could not have**: 84.85 at `k = 1`; 86.96 and 78.26
at two; 85.00, 65.00 and 55.00 at three; 82.35, 64.71, 52.94 and 47.06 at four —
identical in both arms of all five pairs, as is the tokens-a-round column, which
is what a change that moves no token has to read as.

### Against the reference, one sitting

`just bench-engines` on the packed heads, three pairs, both engines alternating,
every row of the four with its ranges apart:

    prompt × answer   ours mean   ours p50   mlx-vlm mean   mlx-vlm p50   ahead on p50
        97 × 128        17.29ms    17.32ms       23.05ms       22.90ms          1.32×
       385 × 128        24.78ms    18.64ms       23.44ms       23.22ms          1.25×
       769 × 128        25.86ms    19.47ms       23.80ms       23.64ms          1.21×
        97 × 512        18.03ms    18.48ms       23.24ms       23.15ms          1.25×

Ahead of D2's 1.30, 1.23, 1.19 and 1.24 at every row. The two long prompts' means
still carry the deferred step — 772 to 881 ms at step 1 — and the medians are the
decode step. Speculating two deep, the same sitting: **15.06 ms a token at 97 ×
128 and 14.81 at 97 × 512 against the reference's 23.05 and 23.24**, which is
1.53× and 1.57×. Both engines wrote the same first eight tokens at every one of
the four prompts.

### What did not move

**No token changed.** All 657 gated cases pass against a real checkpoint and all
66 of the timing tier pass after them, `--backend cpu` included; the recorded
continuation `[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both
backends write. The counts are 646 and 61 plus the eleven gated and five timing
cases these three commits and the probe above add. The numerics flag stays
defaulted to reference.

**And it is the same continuation through a rewind at every depth**, which is
the one thing on this list a merged dispatch could break quietly. The
convolutions carry a window between calls and a rejected guess reaches back into
it, so a pair that swapped the wrong one would answer this step correctly and
every step after it wrongly.
`speculating_writes_the_text_that_decoding_one_token_at_a_time_writes` now drives
`k` of 1, 2, 3 and 4 where it drove 1, 2 and 4 — a depth left out is a rewind
nobody drove — and every one of them rejects guesses on the way through: 2 of 3
accepted at one, 2 of 6 at two, 2 of 8 at three, 2 of 10 at four, and the
oracle's own text out of all five runs.

**Prefill did not move.** Three pairs each, against the commit before the merges:

    tokens     wall    device   ranges
     2048     -0.1%     +0.0%   across, no claim
     4096     +0.1%     -0.0%   across, no claim
     8192     +0.2%     +0.0%   across, no claim

Which is what a decode-step lever should read as: a prefill's dispatches are
hundreds of rows each, so 124 launches out of a call that already runs for
eleven seconds is four decades under its own noise.

**K1's kept cache is untouched**: a simulated coding session, three pairs, every
pair the same way and the ranges apart, reads **71.20 s kept against 265.23 s
cold** — every turn prefilling the same 321 tokens where the cold arm re-prefills
9472 — and the per-turn bookkeeping is **1.143 ms**, inside the 1.013 to 1.267
this file has at the two ends of a context sweep. It is a kept-against-cold
sitting rather than a before-against-after one: what it says is that the
arrangement is where it was.

**Both MMA floors,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert`,
`the_bounded_loop_is_the_unbounded_one_bit_for_bit` and
`a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine` are where
they were**, and so is
`a_paired_dispatch_answers_what_the_two_dispatches_it_replaces_answer`.

### The floor this stops at, and what is under it

**A decode step still reads 5.91 GB of packed weight once, and at 725 GB/s that
is 8.15 ms.** The step is at **51% of it on the device where it was at 50%**,
over the paired sitting above.

    term                                        cost   what it is
    the weights, at the bus                   8.15ms   the floor
    876 launches, at 2.60µs each              2.28ms   being 876 dispatches
    everything else                           5.54ms   the walk, and the bytes above the floor

**The second row is what the next lever has to be, and it is no longer a count.**
Three milestones have taken the thousand to 876 by merging what could be merged,
and what is left does not merge: the dispatches a step makes now are the model's
own shape. What is under them is the ordering — `what_the_device_makes_of_a_
thousand_indirect_commands` reads a barrier at **91% of what a dispatch costs the
device when it computes nothing**, measured this milestone from inside a
mechanism that can address one command at a time, where D2 had it as an estimate.

It stays left, for D2's reason and now with a number on it: it puts a wrong
answer one missing barrier away with nothing in the arithmetic to catch it.

**It did not stay left, and the number above is the one that had to be
re-measured.** See "The ordering under the dispatches": the ordering came off,
the barriers that remain are derived rather than judged, and a decode step is
6.2% shorter on the device for it. What this paragraph got right is the risk;
what it got wrong is that 91% of an *empty* dispatch is the whole cost. On
dispatches that compute something the ordering is worth six times that, because
what it costs is the execution it prevents from overlapping rather than the
launch it delays.

## A decode step's thousand dispatches

**A decode step made 1080 dispatches and it makes 1000.** The eighty are every
MoE bank's `gate` and `up`, which were two dispatches because they are two
tensors and for no other reason: they read the same rows of the same input
through the same expert list against different weights, and neither reads
anything the other writes. Fused, the packed matmul's dispatches go **457 to
377** — the 623 that are not the matmul are where they were — and the device's
own clock **16.718 ms to 16.386**, seven alternating pairs, every pair the same
way and the ranges apart, against a null pair that read −0.1% and a host settled
to 0.5% on the wall before the sitting.

**The step's wall did not follow it, and that is the second finding.** 17.501 ms
to 17.445: seven of seven pairs the same way and the two ranges touching at a
thousandth of a millisecond, which by this file's own standard is no claim at
all. Why is below, and it is what decides what the next lever has to be.

### What a dispatch costs, which is six times what its launch costs

"What a decode step waits on" below priced the 623 dispatches that are not the
matmul at **about 1.1 ms**, out of the 1.4 to 2.0 microseconds `short_conv`'s own
empty-dispatch table measures a launch at. **That is the launch and it is not the
dispatch.** The same profile row those 623 come out of says both what they cost
and what they move:

    the 623 dispatches that are not the matmul     6.59ms    146.25 MB
    what the bus would take to move 146.25 MB      0.20ms
    what is left, over 623 dispatches              6.39ms     10.3µs each

**So 97% of what a step's non-matmul dispatches cost is not their bytes**, and a
launch is a sixth of what is left. `rms_norm` is 169 of them for 5.94 MB: 11.2
microseconds a call to move 35 KB, which the bus does in 0.05. The rest is what
the reduction sweep already named from the other side — a dispatch costs about
4.4 microseconds before it reads anything, and then walks a loop whose trip count
is a runtime value with nothing to unroll it.

**Which is what the fusion is worth and why.** Eighty dispatches came out of a
step for 0.332 ms of device time, and that is **4.15 microseconds a dispatch** —
the 4.4 the reduction sweep measured for a call that reads nothing at all, to
within the clock either was taken on. A merge removes no work. It removes one of
the two fixed costs the work was carrying.

### The pair a bank's gate and up are

**One grid of twice the output elements, the first half against one weight and
the second against the other.** The untiled entry is emitted twice out of one
template now, the way the tiled one already was, and what the second emission
takes beside the first's bindings is a second bank's codes, its scales, its
output, and the two offsets its bytes begin at. Everything else about the call —
the rows, the widths, the expert list, the strides — describes both banks, and
`PackedPair::new` is where the three shapes that have to agree for that to be
true are made to.

**It keeps the bits by construction rather than within a tolerance.** An output
element is still one simdgroup's `simd_sum` over lanes walking one weight row in
`BYTES_PER_LANE` chunks from the same byte in the same stride. What moved is
which row the walk is pointed at, and that is decided once above the loop out of
an index every lane of a simdgroup shares — so a simdgroup takes one side of it
whole and there is no divergence to pay for.
`a_paired_dispatch_answers_what_the_two_dispatches_it_replaces_answer` holds the
fused answer against the two dispatches it replaces, exactly, at the routed
bank's layout, at the shared bank's, and at the grouped one it does not fuse.

**And it costs the pipeline nothing**, which the same case asserts beside what it
answers: a threadgroup of either entry is that many independent simdgroups with
no threadgroup memory at all, so the widest threadgroup this device gives the
paired entry is the widest it gives the plain one. A merge that had halved the
occupancy would be giving width back on the axis it was taken for.

**The width is the other half of what it buys and it is the larger half.** The
rate curve this kernel was measured along last milestone runs from 12% of the bus
at 512 output elements to 95% at 131072, and every shape a decode step has is on
the steep part of it: a routed bank's pair goes from 12288 elements to 24576 and
a shared bank's from 4096 to 8192. In the model, over the same 5932.36 MB:

    entry                calls     device        moved   achieved   of bus
    packed_matmul          457    13.66ms   5932.36 MB   434 GB/s      60%
    packed_matmul          297     7.50ms   3072.37 MB   409 GB/s      56%
    packed_matmul_pair      80     5.41ms   2859.99 MB   529 GB/s      73%
    both, after            377    12.91ms   5932.36 MB   460 GB/s      63%

The last row is the two above it added up rather than a row the instrument
printed, which is why its rate is arithmetic where theirs are measured.

**Both figures carry the sampling bias that table declares** — the passes sum to
19.51 ms where the same step unsampled is 15.94 — and are read against each other
rather than absolutely. The paired row is 73% of the bus where the untiled row it
came out of is 56%.

**A grouped call is left as two dispatches and so is a tiled one.** Those are a
prefill's, where a dispatch already has output elements enough to fill the
machine — a 769-token prefill's routed gate is 9.4 million of them — and what a
tile carries is registers, on the entry where they are already the binding
constraint. There is no launch worth noticing beside that walk and no width left
to win.

### Why the device moved and the step did not

**A decode step's wall is not its device time, and the millisecond between them
is this process.** The round-trip table divides a submission into what the driver
took to make it runnable, what it then spent queued, and what it executed for —
one row per shape of submission, the `a step` column being how many of that shape
a step makes and every duration the sum over them:

    a 18.22ms step, of which 10.43ms is the 14 submissions it waits for
    dispatches   a step     waited  scheduled    queued   executed
    54                1     1.52ms    94.79µs    8.65ms     1.46ms
    75               12     8.79ms     1.85ms   56.07ms    13.85ms
    88                1     2.00µs   133.04µs  432.75µs     1.47ms

So a step is 18.22 ms of which 10.43 is spent inside a wait and **the remaining
7.79 is this process encoding the next command buffer while the GPU runs the last
one**. The two
overlap and the wall is neither: it is the longer of them per submission, plus
whatever neither covers. Eighty fewer dispatches took 0.33 ms off one side and
about 0.3 ms off the other, and the wall moved 0.06.

**What that says about the next lever is the useful part.** Removing a dispatch
pays twice — once on the device and once in the encode that has to stay ahead of
it — and at this ratio neither payment reaches the wall on its own. It is the
count that has to come down far enough for both, and eighty of 1080 was not far
enough.

### What is left, priced

**Three pairs of dispatches a decode step makes are independent of each other and
could each be one dispatch**, by the change this milestone just made at another
kernel:

    what                                  merges   dispatches
    a layer's query norm and key norm         42           42
    a layer's key and value convolutions      42           42
    a layer's routed and shared SwiGLU        40           40

**124 of the 623, and worth 0.5 to 1.1 ms of a 15.9 ms device step.** The low end
is what this milestone measured for the merge it made, 4.15 microseconds a
dispatch removed; the high end is a whole one of these dispatches less its bytes,
which for `rms_norm` is 9.2 microseconds unsampled. **Nothing in that range is
available from a launch alone**, which is the reading the 1.1 ms above was made
under.

**None of the three costs threadgroup memory and none changes a summation
order.** They are elementwise operations or short reductions writing separate
outputs, so a merged dispatch is the same arithmetic over a grid that names which
half a simdgroup is in — the pair above, at another kernel. What they cost is a
second set of bindings each, and for the SwiGLU a MoE layer that dispatches both
banks' pairs before either activation rather than finishing one bank first.

**And what is under all of it is not reached here.** Metal's default dispatch
type is serial: consecutive dispatches in one encoder are separated by a barrier
the driver inserts, and that barrier is what most of the 4.4 microseconds is. An
encoder made concurrent, with barriers written only where a dependency is real,
would take it off every independent pair at once rather than one kernel at a
time — and would put a wrong answer one missing barrier away, with nothing in the
arithmetic to catch it. It is named and priced here and left.

### The step after a prefill, which is not a decode step

**`Asked::settle` has thrown one step away since the cross-engine table found it,
and what that step is was a sentence rather than a measurement.** The
cross-engine table records it at 736 and 783 ms and loses two of its four rows'
means to it; a user waits for it either way.

`what_the_step_after_a_prefill_is_paying_for` prices it. **It is not work**: it
dispatches what the step behind it dispatches, submits what it submits and
allocates what it allocates, to the dispatch and the mebibyte. **It is not a
compilation**: every pipeline this engine has was created before the prefill ran.
And of the 840.94 ms below, the device executes for 20.17 — the three rows' own
`executed` column, summed.

**It is scheduling** — the driver turning committed buffers into work the GPU can
start:

    a 840.94ms step, of which 834.47ms is the 14 submissions it waits for
    dispatches   a step     waited  scheduled    queued   executed
    54                1     2.18ms     2.50ms    74.12µs    1.62ms
    75               12   792.75ms   793.07ms    7.42ms    16.82ms
    88                1    39.43ms    42.77ms  589.29µs     1.73ms

The twelve middle submissions of this one step are **793 ms scheduled against 17
ms executed**, where the same twelve of the step behind it are 1.9 ms scheduled.

**And it is a threshold rather than a slope**, which is why the cross-engine
figures at 385 and 769 tokens are so nearly equal:

    context   prefill    step 1    step 2    step 3   allocated
         97     1.10s   18.16ms   18.08ms   18.22ms    1687 MiB
        193     1.89s  487.80ms   18.20ms   18.18ms    3385 MiB
        385     2.82s  807.81ms   19.03ms   19.44ms    6873 MiB
        769     4.94s  769.29ms   19.96ms   20.18ms   13741 MiB
       1537     8.46s  815.74ms   21.38ms   21.36ms   27467 MiB

**Nothing at 97 tokens, six tenths of it at 193, and about 0.8 s from 385 on.**
What saturates is what the reading rests on: the cost stops growing while the
prompt doubles twice more and while what the prefill allocated goes from 6.9 to
27.5 GiB, so what the step is re-taking is a working set of its own rather than
anything the prefill left proportional to itself. **A decode step's is 5.9 GB of
packed weight and 5.9 GB in 0.8 s is 7.4 GB/s — arithmetic, and the order of a
rate this engine has measured for a checkpoint arriving from disk rather than for
one already resident.** Which of the driver's several residency structures it is
this side cannot see, and the row above is what can be said without seeing it.

**What it is not is free to move off the path.** It is a cost the driver pays
once per prefill, and this side can choose when it is paid rather than whether:
the same 0.8 s charged to the prefill's own row is the same 0.8 s in the wall a
user waits. What would remove it is a prefill that does not churn 13.7 GiB of
allocations in the first place, which is a different milestone's lever. What it
does settle is that a table quoting only a mean is quoting the deferral as the
step's price — which is why `just bench-engines` prints both and why
`Asked::settle` takes it out of every other decode figure here.

### What a decode step costs now, at every context

`what_a_decode_step_costs_as_the_context_grows`, one sitting each side:

    context     before      after   tokens/s   of floor   device before   device after
       97      18.16ms    18.06ms  55.1→55.4    45%→45%        16.97ms        16.68ms
      385      19.49ms    19.37ms  51.3→51.6    42%→42%        18.28ms        18.01ms
      769      20.19ms    20.04ms  49.5→49.9    40%→41%        19.02ms        18.69ms
     2048      23.01ms    22.82ms  43.5→43.8    35%→36%        21.74ms        21.45ms
     4096      24.37ms    24.03ms  41.0→41.6    33%→34%        23.17ms        22.61ms
     8192      27.20ms    26.86ms  36.8→37.2    30%→30%        25.95ms        25.47ms

**`of floor` is 8.15 ms over the step**, which is what a step would be if the
weights arrived at the bus's own rate and everything else were free. The mean and
the median agree to a twentieth of a millisecond at every row — 18.06 against
18.13, 26.86 against 26.92 — which is what says no row has a step in it that is
not a decode step, the section above being about the one that is not.

### What speculation costs now, which is the same tokens at every depth

Five alternating pairs on the packed heads:

    depth    a token before   a token after   tokens/s   device before   device after
    k = 0         17.743ms        17.637ms   56.4→56.7        16.885ms        16.548ms
    k = 1         15.677ms        15.671ms   63.8→63.8        14.606ms        14.458ms
    k = 2         15.378ms        15.372ms   65.0→65.1        14.097ms        13.986ms
    k = 3         16.270ms        16.343ms   61.5→61.2        14.650ms        14.645ms
    k = 4         18.698ms        18.776ms   53.5→53.3        16.664ms        16.698ms

**The device row is a claim at `k` of 0, 1 and 2 and no claim at 3 and 4, and the
kernel predicts exactly that.** A round of depth `k` puts `k + 1` rows through
every bank, and a shared bank's rows are the input laid end to end once per
shared expert — so at `k` of 3 its runs reach `ROWS_A_TILE` and the pair goes
down the tiled entry, which this milestone deliberately did not fuse. Half the
merge stops applying at exactly the depth the arithmetic says it should. The
wall row is no claim at any depth, for the reason the previous section gives.

**Acceptance did not move and could not have**: 84.85 at `k = 1`; 86.96 and 78.26
at two; 85.00, 65.00 and 55.00 at three; 82.35, 64.71, 52.94 and 47.06 at four —
identical in both arms of all five pairs, which is what a change that moves no
token has to read as. The depth that pays best is still two.

### Against the reference, one sitting

`just bench-engines` on the packed heads, three pairs, both engines alternating,
every row of the four below with its ranges apart:

    prompt × answer   ours mean   ours p50   mlx-vlm mean   mlx-vlm p50   ahead on p50
        97 × 128        17.72ms    17.70ms       23.17ms       23.04ms          1.30×
       385 × 128        24.83ms    18.99ms       23.32ms       23.32ms          1.23×
       769 × 128        26.30ms    19.82ms       23.86ms       23.68ms          1.19×
        97 × 512        18.46ms    18.81ms       23.34ms       23.27ms          1.24×

**The two long prompts' means still carry the deferred step** — 735 to 851 ms at
step 1, which the section above is about — and the medians are the decode step.
Speculating two deep, the same sitting: **15.23 ms a token at 97 × 128 and 14.89
at 97 × 512 against the reference's 23.17 and 23.34**, which is 1.52× and 1.57×.
Both engines wrote the same first eight tokens at every one of the four prompts.

### What did not move

**No token changed.** All 646 gated cases pass against a real checkpoint and all
61 of the timing tier pass after them, `--backend cpu` included; the recorded
continuation `[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both
backends write. The numerics flag stays defaulted to reference and the production
entries are untouched — this milestone changed the entry neither of them stands
in front of.

**Prefill did not move**, which the compile order in `PackedMatmul::compiled` is
what protects: the paired entry is created where the untiled one is, after both
tiled entries. Three pairs each, against the commit before the pair existed:

    tokens     wall    device   ranges
     2048     -0.2%     +0.0%   across, no claim
     4096     +0.2%     -0.0%   across, no claim
     8192     -0.3%     -0.0%   across, no claim

**K1's kept cache is untouched**: a simulated coding session, three pairs, every
pair the same way and the ranges apart, reads 71.86 s kept against 264.8 s cold —
every turn prefilling the same 321 tokens where the cold arm re-prefills 9472 —
and the per-turn bookkeeping is **1.094 ms**, the flat-in-the-context figure that
is this engine's one measured architectural advantage, against the reference's 31
to 66. It is a kept-against-cold sitting rather than a before-against-after one:
what it says is that the arrangement is where it was, and the 1.094 ms is the same
figure this file has at 1.013 and 1.267 at the two ends of a context sweep.

**Both MMA floors,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert`,
`the_bounded_loop_is_the_unbounded_one_bit_for_bit` and
`a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine` are where
they were.**

### The floor this stops at, and its arithmetic

**A decode step still reads 5.91 GB of packed weight once, and at the 725 GB/s
`what_a_streaming_read_achieves_on_this_machine` measures that is 8.15 ms.** The
step is at 50% of it on the device where it was at 49%, over the paired sitting
above.

Where it goes is the profile's shares divided into that run's own unsampled step,
which is 15.94 ms — a different sitting from the paired 16.386 and a ratio rather
than a duration, so the three terms are read against each other:

    term                                        cost   what it is
    the weights, at the bus                   8.15ms   the floor
    the matmul above the floor                 2.40ms  377 dispatches and a walk
    every other kernel in a step               5.39ms  623 dispatches moving 146 MB

**The matmul is at 77% of its own floor where it was at 73%**, which is the same
division of the run before it: 68.1% of that step's passes against 66.2% of this
one's, over unsampled steps of 16.46 ms and 15.94. What the third row does *not*
show is a change — it reads 5.25 ms in the run before and 5.39 here, which is the
two sittings and not the 623 dispatches, none of which this milestone touched.

**The third row is the one this milestone leaves as the largest.** It is 146 MB
of traffic, which the bus would move in 0.20 ms, and 623 dispatches, which at
their fixed cost are the rest of it. **A dispatch's fixed cost is 4.15
microseconds by this milestone's own measurement and a step issues a thousand of
them**, so **4.2 ms of a 16.4 ms step is what a step pays for being a thousand
dispatches** rather than for anything any of them computes. That is the stopping
condition and it is a count rather than a kernel: the three merges above take the
thousand to 876, and the arithmetic says what each one is worth before it is
written.

## What a decode step waits on

**A decode step reads the model once and the bus can do that in 8.15 ms. It took
19.44 and it now takes 17.74** — device 18.63 to 16.93, seven alternating pairs,
every pair the same way and the ranges apart, against a null pair that read −0.1%
and a host settled to 0.39% on the wall before the sitting.

**What was in the way was not the arithmetic and not the bytes. It was that a
lane waited a memory round trip for four bytes before it asked for four more**,
and that a decode step's dispatches are too narrow to hide that wait behind
anything else.

### The kernel a decode step actually runs, which nothing here had measured

`which_kernels_own_a_decode_step` puts `packed_matmul` at 457 calls and 68% of
the device's time in a step. **It is the untiled entry — one simdgroup an output
element — and it is the one kernel in this crate no milestone has pointed an
instrument at.** Both entries behind the numerics flag refuse a call
shorter than 64 rows, `tiles` is false for every shape a step has, and the
occupancy turn A5 took declares memory only on the tiled entries. A7's and A8's
floors exclude a one-row dispatch by construction. What runs is the code this
engine shipped before any of that work.

`what_each_shape_of_a_decode_steps_packed_calls_costs` is that row divided up. A
step's ten shapes, their call counts the engine's own, warm, swept both ways with
the banks uploaded once and held across both passes:

    shape               elements    weight     input    a call   achieved  of bus    a step
    q_proj, o_proj          4096    8.9 MB     67 MB      23µs   388 GB/s     54%    1.93ms
    k_proj, v_proj          1024    2.2 MB     17 MB      14µs   160 GB/s     22%    1.17ms
    r_proj                   512    1.1 MB      8 MB      13µs    86 GB/s     12%  546.00µs
    shared gate, up         4096    8.9 MB     67 MB      23µs   393 GB/s     54%    1.81ms
    shared down             8192    8.9 MB     67 MB      21µs   431 GB/s     59%  827.08µs
    routed gate, up        12288   26.7 MB    201 MB      49µs   550 GB/s     76%    3.89ms
    routed down            24576   26.7 MB    201 MB      48µs   556 GB/s     77%    1.92ms
    dense gate, up         16384   35.7 MB    268 MB      61µs   582 GB/s     80%  245.16µs
    dense down              4096   35.7 MB    268 MB      75µs   475 GB/s     65%  150.23µs
    lm_head               200058  435.3 MB   3278 MB     627µs   694 GB/s     96%  627.12µs

**The `a step` column sums to 13.13 ms against the profile's 15.10**, which is
what says the decomposition is of the row rather than beside it — the profile's
figure carries the sampling bias its own table declares. The weight totals 5.91
GB, which is the checkpoint's own arithmetic arriving from a dispatch's
declaration: **at 725 GB/s that is 8.15 ms and it is the floor.**

**The `elements` column is what the table turns on and the rate column follows
it**, from 86 GB/s at 512 output elements to 694 at 200058. Every shape a step
dispatches is in the first half of that range and the head is the only one in the
second.

### Whether a narrow dispatch is short of the machine, which is yes

K2 left the hypothesis in one line and no milestone acted on it: *"the laggards
are dispatches with 1024–4096 output elements, too few to fill 80 cores."* An
output element is one simdgroup here, so
`whether_a_decode_dispatch_is_short_of_the_machine` asks it directly — the same
one-row call at nine widths, warm, both ways:

    elements   weight    a call   achieved  of bus
         512   1.1 MB      13µs    85 GB/s     12%
        1024   2.2 MB      13µs   168 GB/s     23%
        2048   4.5 MB      17µs   270 GB/s     37%
        4096   8.9 MB      22µs   397 GB/s     55%
        8192  17.8 MB      35µs   514 GB/s     71%
       16384  35.7 MB      61µs   588 GB/s     81%
       32768  71.3 MB     113µs   631 GB/s     87%
       65536 142.6 MB     212µs   674 GB/s     93%
      131072 285.2 MB     415µs   687 GB/s     95%

**The kernel reaches 95% of the bus and a decode step never gives it a call wide
enough to.** Its widest bank is 24576 elements and its narrowest projection is
512, and the row above fits — to within the clock — to a fixed cost plus the
bytes at 720 GB/s.

**Every one of those figures is warm and both of those words are load-bearing.**
The same sweep unwarmed reads 49, 93, 153, 223, 310, 350, 378, 393 and 683 GB/s —
the same monotone shape and almost none of it the kernel. It is the third time
`testing::warmed` has caught this in this repo and the first where the row it
drew was the row the milestone was about.

### What the fixed cost is, which is one simdgroup's own walk

Two more sweeps, because a constant that comes out of a fit is a constant that
has to be attributed. `what_a_decode_dispatch_costs_beside_more_of_its_own_kind`
puts more dispatches in one command buffer, which divides a per-buffer cost away
and leaves a per-dispatch one alone:

    a buffer   a dispatch  achieved
           1       16.8µs  133 GB/s
           4       14.3µs  156 GB/s
          16       13.4µs  167 GB/s
          64       13.4µs  166 GB/s

**A command buffer is 2.6 microseconds on the device and the rest does not
divide away.** `what_a_decode_dispatchs_fixed_cost_is_made_of` then sweeps the
reduction at a fixed width, so that what changes is how far one simdgroup walks
and not how many of them there are:

    reduction   steps   weight    a call  at the bus  the rest
          512       2   0.3 MB     5.1µs       0.4µs     4.7µs
         1024       4   0.6 MB     6.8µs       0.8µs     6.1µs
         2048       8   1.1 MB     8.9µs       1.5µs     7.4µs
         4096      16   2.2 MB    14.0µs       3.1µs    10.9µs
         8192      32   4.5 MB    23.2µs       6.1µs    17.1µs
        16384      64   8.9 MB    42.7µs      12.3µs    30.4µs

**A dispatch costs 4.4 microseconds plus 0.41 a step, and 0.41 microseconds is a
memory round trip.** A lane covers `BYTES_PER_LANE` bytes a step and a simdgroup
32 of those, so a reduction of 4096 is sixteen steps whatever the call's width —
and the walk's trip count is a runtime value, so nothing unrolls it. A lane
issued four bytes, waited for them, added them and issued four more. **Across the
457 calls a decode step makes that is 2.75 ms of an 18.6 ms step spent waiting
rather than reading.**

### What the walk waits on, one term at a time

A4 gave the reference tile a limiter table and A9 gave the block one. Both are
kernels a prefill dispatches and a decode step never reaches.
`what_a_decode_steps_packed_matmul_is_bound_by` is the same instrument on the
walk that runs 457 times a step — **two arms a term, because confining a walk to
a kilobyte leaves every load issued and takes only the traffic off, where taking
the load out prices the load as well:**

    without                                q_proj        k_proj    a routed bank
    nothing — the kernel                23µs  100%    14µs  100%    52µs  100%
    the weight's traffic                23µs  100%    16µs  119%    50µs   96%
    the weight's loads                  18µs   75%    11µs   79%    38µs   73%
    the input's traffic                 18µs   77%    11µs   81%    37µs   71%
    the input's loads                   16µs   67%    11µs   79%    30µs   57%
    the table it decodes through        19µs   82%    11µs   79%    41µs   78%
    the scale it walks to               23µs  100%    14µs  104%    48µs   91%
    the cross-lane reduction            40µs  172%    24µs  178%    71µs  137%

**Every term that is a load is worth a fifth to two fifths, and every term that
is only bytes is worth nothing.** Confining the whole weight working set to one 2
KB row is 96 to 119% — it is *slower*, because 4096 simdgroups then hammer one
cache line — and the group scale is worth nothing at all. The input is the term
this walk re-reads hardest: every output element is a simdgroup of its own and
every one reads the whole input row, which is 7.5 bytes of input load issued for
every byte of weight at every shape a step has. But confining that traffic buys
23%, where removing the loads buys 33%. **The walk is waiting on loads and not on
bandwidth**, and that is the same finding the width sweep makes from the outside.

**The last row did not reproduce and is printed rather than dropped.** Two other
sittings of the same arm put it at 100 and 101% on all three shapes, where this
one has it 37 to 78% *slower* than the kernel it removes work from. What it
prices is a compiler doing something else with a value nobody reduces; what the
reduction itself is worth is nothing, which all three sittings agree on from
opposite directions.

### The two things this changes, and both keep the bits

**A lane reads four chunks of its weight row before it adds any of them.** The
chunks are read in the order the walk visits them and added in that order, so the
sum is the same expression associated the same way — this is a constant and not a
second numerics, which is what separates it from every other lever this module
has taken on the innermost walk.
`a_walk_that_holds_several_chunks_answers_the_bits_one_holding_one_answers` holds
it against the shipped kernel's own bit patterns at five chunk counts, over every
reduction the checkpoint has and two that divide neither the lane stride nor any
count above one.

**Four, and the sweep is in the model.** Seven alternating pairs of a decode step
with the ranges apart at every arm: two held is 0.9% behind four and eight is
1.8% behind, at both threadgroup widths this module has shipped.
`what_a_packed_multiply_costs_at_each_chunk_a_lane_holds` asks it of one dispatch
at a time and turns at the same place — 1.43× at `k_proj`, 1.19× at a shared
bank, 1.08× at a routed one and **1.02× at a call wide enough to hide its own
loads**. That last column is the finding rather than the first: what a held chunk
buys is buying a narrow dispatch the latency hiding a wide one already had.

**And an untiled dispatch gets a threadgroup of its own, which is 64 threads.** A
threadgroup of this entry is that many independent simdgroups and nothing else —
no threadgroup memory, no barrier, nothing shared — so its width reaches the grid
and stops there, and every arm of a sweep over it answers the same bits. What it
decides is how evenly a call's simdgroups land on 80 cores: at the shipped 256 a
1024-element dispatch is 128 threadgroups of eight, so 48 cores take two and 32
take one and the dispatch waits for the slower half.

    a threadgroup      a decode step   the device
     32 threads             -2.1%         -2.2%
     64 threads             -2.3%         -2.3%
    128 threads             -1.8%         -1.9%
    256 threads, before         —             —
    512 threads             +3.5%         +3.8%

**The middle of a plateau rather than its edge**, which is the discipline
`RESIDENCY` is sized under: seven pairs each of 64 against 32 and against 128
read +0.1% and +0.2% with the ranges across and no claim, so the three are one
arm. It is its own constant because `THREADS_PER_GROUP` is not free to move —
that one is the tiled entries' as well and a block's shape is asserted against
it, so a sweep that turned it turned the prefill's shape too, and the two want
opposite widths.

### The order three pipelines are created in, which is worth 2.4% of a prefill

**This is the strangest thing this milestone found and it is a measurement rather
than an explanation.** With the held walk in place, a 2048-token prefill cost
2.4% of its device time and an 8192-token one 2.2% — on a kernel the sampled
profile puts at one call and 0.0% of a prefill. The cost was proportional to the
prompt, so it was the two *tiled* entries running slower, which are 87% of a
prefill.

It is not the source. Compiling the untiled entry from a string of its own, so
that the tiled two never saw the walk it runs, left the 2.4% exactly where it
was. **What moved it was which of the three pipelines was created first**: moving
`packed_matmul`'s `device.compile` after the two tiled ones gives all of it back
and 0.5% besides. Nothing about the three says this should be so — they are three
libraries out of one source string, dispatched at different shapes with nothing
shared between them. It is written down in `PackedMatmul::compiled` because the
next edit to that function could undo it by accident.

### What a decode step costs now, at every context

`what_a_decode_step_costs_as_the_context_grows`, one sitting each side, the same
build either way but for the two constants:

    context     before      after   tokens/s   of floor   device before   device after
       97      20.00ms    18.16ms  50.0→55.1   41%→45%         18.77ms        16.97ms
      385      22.47ms    19.49ms  44.5→51.3   36%→42%         21.28ms        18.28ms
      769      22.01ms    20.19ms  45.4→49.5   37%→40%         20.75ms        19.02ms
     2048      24.89ms    23.01ms  40.2→43.5   33%→35%         23.69ms        21.74ms
     4096      26.28ms    24.37ms  38.1→41.0   31%→33%         24.95ms        23.17ms
     8192      28.91ms    27.20ms  34.6→36.8   28%→30%         27.65ms        25.95ms

**`of floor` is 8.15 ms over the step**, so it is what a step would be if the
weights arrived at the bus's own rate and everything else were free. The mean and
the median agree to a tenth of a millisecond at every row, which is what says no
row has a step in it that is not a decode step.

The per-shape table re-taken through the same instrument on the same host:

    shape               elements   before      after   of bus
    q_proj, o_proj          4096  388 GB/s   484 GB/s  54%→67%
    k_proj, v_proj          1024  160 GB/s   264 GB/s  22%→36%
    r_proj                   512   86 GB/s   140 GB/s  12%→19%
    shared gate, up         4096  393 GB/s   494 GB/s  54%→68%
    shared down             8192  431 GB/s   502 GB/s  59%→69%
    routed gate, up        12288  550 GB/s   621 GB/s  76%→86%
    routed down            24576  556 GB/s   592 GB/s  77%→82%
    dense gate, up         16384  582 GB/s   634 GB/s  80%→87%
    dense down              4096  475 GB/s   602 GB/s  65%→83%
    lm_head               200058  694 GB/s   469 GB/s  96%→65%

**and the ten shapes are 13.13 ms of a decode step against 11.27** — 62% of the
floor against 72%. **The head's row is the one not to read**: its two directions
disagreed by 13% where eight of the ten agreed inside 6%, and
`one_dispatch_does_an_lm_head_shaped_multiply_without_meeting_the_watchdog` —
which dispatches the checkpoint's own head cold and times it on a wall clock —
reads 303 GB/s before and 297 after. The head did not move.

In the model, over the same 457 calls and the same declared 5932.36 MB:
`packed_matmul` is **15.10 ms and 393 GB/s before, 13.66 ms and 434 GB/s
after** — 54% of the bus against 60%. **Both of those carry the sampling bias
that table declares** and are read against each other rather than absolutely: the
passes sum to 20.09 ms where the same step unsampled is 16.93, so a row is about
19% over. The floor arithmetic below divides the *share* into an unsampled step
rather than quoting the row, which is 68.0% of 16.93 ms and 11.5.

**And 5932.36 MB is not the 5.91 GB the floor divides.** A dispatch declares the
rows it reads and writes beside the weight it walks, so the profile's figure is
22 MB of activations more than the codes and scales a token must fetch. The floor
is on the weight.

### What speculation costs now, which is faster at every depth and narrower at all of them

Five alternating pairs on the packed heads, every pair the same way and the
ranges apart at every row:

    depth    a token before   a token after   tokens/s   speedup before   speedup after
    k = 0         19.439ms        17.736ms   51.4→56.4            1.000           1.000
    k = 1         16.556ms        15.803ms   60.4→63.3            1.174           1.122
    k = 2         16.010ms        15.418ms   62.5→64.9            1.214           1.150
    k = 3         16.503ms        16.338ms   60.6→61.2            1.178           1.086
    k = 4         19.462ms        18.793ms   51.4→53.2            0.999           0.944

**Every depth is faster in tokens a second and every ratio is narrower, and the
second of those is the first arriving as a denominator.** A speedup here divides
by that run's own `k = 0`, and `k = 0` moved 8.8% — so a round that banks the
same tokens for the same work reports a smaller multiple. The depth that pays
best is still two and nothing about which one to run has changed.

**Acceptance did not move and could not have**: 84.85 at `k = 1`; 86.96 and 78.26
at two; 85.00, 65.00 and 55.00 at three; 82.35, 64.71, 52.94 and 47.06 at four —
identical in both arms of every pair, which is what a change that moves no token
has to read as.

### Against the reference, one sitting

`just bench-engines` on the packed heads, three pairs, both engines alternating,
ranges apart at every row. **The mean and the median are both printed because
they disagree and the reason is ours:**

    prompt × answer   ours mean   ours p50   mlx-vlm mean   mlx-vlm p50   ahead on p50
        97 × 128        17.83ms    17.83ms       22.96ms       22.82ms          1.28×
       385 × 128        24.83ms    19.11ms       23.33ms       23.10ms          1.21×
       769 × 128        26.10ms    20.00ms       23.72ms       23.55ms          1.18×
        97 × 512        18.58ms    18.92ms       23.12ms       23.03ms          1.22×

**What separates our mean from our median is one step of about 780 ms**, and it
is the first step after a prefill of 385 or 769 tokens — work the prefill defers
onto it. On the mean this engine loses those two rows by 6 and 9%; on the median
it leads all four by 1.18 to 1.28×. Both are true and the deferred step is real
work a user waits for; what it is not is a decode step, and a table that quoted
only the mean would be reporting the deferral as the step's price. `Asked::settle`
is what takes it out of every other decode figure in this file.

Speculating two deep, the same sitting: **15.25 ms a token at 97 × 128 and 14.95
at 97 × 512, against the reference's 22.96 and 23.12** — 1.51× and 1.55×. Both
engines wrote the same first eight tokens at every one of the four prompts.

### What did not move

**No token changed.** All 645 gated cases pass against a real checkpoint and all
60 of the timing tier pass after them, `--backend cpu` included; the recorded
continuation `[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both
backends write. The numerics flag stays defaulted to reference and the production
entries are untouched — this milestone changed the entry neither of them stands
in front of.

**Prefill did not move, and the compile order above is why it did not.** Paired
against the commit before the kernel changed, three pairs each:

    tokens    wall     device   ranges
     2048    -0.5%      -0.5%   apart, 3 of 3
     4096    -0.2%      -0.3%   across on the wall, apart on the device
     8192    -0.2%      -0.0%   across, no claim

**K1's kept cache is untouched.** A simulated coding session, three pairs: the
whole session 72.49 s to 71.71 s (−1.1%, ranges apart), turn three −4.4%, and the
per-turn bookkeeping **1.160 ms against 1.169** — ranges across, no claim, which
is the flat-in-the-context figure that is this engine's one measured
architectural advantage. Every turn prefilled the same 321 tokens in both arms.

**Both MMA floors,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert`,
`the_bounded_loop_is_the_unbounded_one_bit_for_bit` and
`a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine` are where
they were.**

### The floor this stops at, and its arithmetic

**A decode step reads 5.91 GB of packed weight — six of each MoE layer's 256
experts and both shared ones, every layer's five projections, two dense feed-forward
networks and the head — and it reads all of it once. At the 725 GB/s
`what_a_streaming_read_achieves_on_this_machine` measures, that is 8.15 ms and
nothing about how the kernel is written can go under it.**

Where the 16.93 ms of device time in a step sits against that:

    term                                        cost   what it is
    the weights, at the bus                   8.15ms   the floor
    the matmul above the floor                 3.4ms   see below
    every other kernel in a step               5.4ms   attention, norms, conv, router, swiglu

The second and third are the profile's own shares of an unsampled step:
`packed_matmul` is 68.0% of the passes a step runs and 68% of 16.93 ms is 11.5,
which is 8.15 of floor and 3.4 above it.

**The matmul is at 71% of its floor and the step is at 48% of it.** The two are
different numbers and the second is the one a user feels: a matmul that ran at
the bus would leave a step at 13.5 ms of device time and 14.3 of wall, which is
**1.24× from here and not 2.1×**.

What the 3.4 ms above the floor is, priced rather than guessed:

- **457 dispatches at about 4 microseconds of fixed cost each is 1.8 ms.** The
  reduction sweep separates this from the walk and it does not fall with the
  walk, so it is what a dispatch costs before it reads anything. **The only thing
  that removes it is removing dispatches.**
- **The rest is the walk still not fully hidden at the narrow shapes.** `r_proj`
  is at 19% of the bus and `k_proj` at 36%; between them they are 1.04 ms of a
  step where the bus would take 0.32.

**What is worth trying next, and what it would cost:**

- **Fusing each bank's `gate` and `up` into one dispatch.** They read the same
  input against different weights, so they are two calls where one would do: 80
  fewer dispatches a step (0.3 ms by the arithmetic above) and, more than that,
  twice the output elements on the two shapes that carry 2.85 GB between them.
  The width curve puts 4096 → 8192 elements at 487 → 580 GB/s and 12288 → 24576
  at 621 → 660, which is another **0.5 to 0.8 ms**. It costs a change to how the
  two tensors are laid out at load, which is the loader and `swiglu` rather than
  the kernel. **A day, and it is the largest thing left.**
- **The 623 dispatches a step that are not the matmul.** `rms_norm` is 169 of
  them for 5.94 MB and `short_conv` 168 for 22.02; between them and the router's
  80 they are 3.9 ms of device time for 2% of a step's bytes. The launch alone,
  at the 1.4 to 2.0 microseconds `short_conv`'s own empty-dispatch table
  measures, is about **1.1 ms**. Nothing here removes a launch but removing a
  dispatch, and these are separate dispatches because they compute separate
  things.
- **Splitting one output element's walk across simdgroups**, which is the
  medicine A1 gave decode's attention. It would hide the rest of the walk at any
  width. **It changes the order the products are summed in**, so it is behind the
  numerics flag or it is nothing — and this milestone's whole argument for
  shipping by default is that neither of its two levers is.

**What this does not license is a claim that decode is finished.** A step is at
48% of a floor that is physics; the two priced items above are worth about 1.5 ms
of the 8.8 between here and it, and the third is behind a flag this milestone
deliberately did not reach for.

## What the matmul was actually running at

**The kernel was never at 24% of anything, and the instrument that said so was
dividing by four.** This milestone opened to make the matmul structurally faster
against an existence proof — mlx at 63.5% of peak where this engine sat at 24%.
Neither number survived. The 24% was a bandwidth roofline over a clock that
under-reported by its own call count; the 63.5% is mlx's *dense* GEMM, and mlx's
quantised MoE matmul — the kernel that is actually this one's counterpart —
turns out to be 1.26× ahead of this one rather than 2.6×.

**What this ships is a correct clock, a denominator that means something, four
percent of kernel, and a floor with arithmetic under it.**

### The clock every table in this crate was read off

`crate::testing::device_time` puts `calls` dispatches in one command buffer and
returns what **one** of them cost — it has already divided. Six places in the
matmul's tests divided by their own `CALLS` a second time, and one multiplied a
bandwidth by it instead. So every absolute figure the block's tables carry is
four times too fast, and the two GB/s columns in the height and width sweeps
disagree with the µs printed beside them.

**Every ratio in those tables stands and none of the absolutes do.** Each arm of
a sweep divided by the same four, so which arm is fastest and what a term is
worth as a percentage are exactly what they were — which is why it survived: a
table read the way a sweep is read looks right. The figures it moves are the ones
a roofline, a percent-of-peak or a cross-engine column divides by, which are the
figures this milestone is about. One dispatch of the routed bank at 8192 tokens
is **54.3 ms and not 13.6**, and `a_blocks_table_reports_one_dispatch_of_the
_shape_it_names` compares against a dispatch measured on its own rather than
against another arm.

#### Which figures in this file inherited it

**Five cases carried it**, and every table in this file that quotes one of them
carries a device time four times too small — and, wherever a rate was formed by
dividing bytes by that time, a bandwidth four times too large:

- `Blocked::costs`, which is the fixture every block table and both of the
  tile's per-shape tables measure through. It reaches **the roofline and limiter
  table, the threadgroup sweep and the 16-bit table** in "The four levers the
  matmul had left", and **the nine packed shapes table** in "Why the two tiled
  rows report bandwidths a factor of two apart".
- `what_separates_a_tiled_call_from_a_grouped_one`, which divided its three time
  columns and *multiplied* its two rate columns — **the expert-count table** in
  that same section, whose µs and GB/s are therefore out by sixteen against each
  other.
- `whether_the_tile_height_turns_at_four_on_a_bank_too_big_to_cache` — **the
  run-length table** in "Whether a long prefill reads one expert's weight more
  than it must".
- `what_a_packed_multiply_costs_at_each_height_a_tile_reads` and
  `..._at_each_width_a_tile_spans`, whose µs was divided and whose GB/s was not.
  Neither table is quoted here.

**There is a rule a reader can apply without the list**, and this file supplies
its own evidence for it: `what_a_streaming_read_achieves_on_this_machine`
measures the bus at **725 GB/s**, and earlier at 819. Every `achieved` column in
this file that reads *above* that — 1079 to 2151 GB/s across the two tables in
"Why the two tiled rows report bandwidths a factor of two apart" — is this bug.
That section diagnosed those rates as denominators inflated by
`PackedBank::moves` charging a whole weight per tile, and that diagnosis is
right and is not the whole of it: **a factor of four of the excess is the
clock.** Divide those rates by four and the amplification argument stands with
nothing physically impossible left in it.

**What is safe, and why:**

- **Every ratio, every percentage and every "which arm won" in every affected
  table.** All the arms divided by the same four.
- **The per-kernel prefill tables** — `kernel / calls / device / moved /
  achieved / of peak`, the ones reading 201 to 666 GB/s. Those come off a
  *sampled* prefill, where the device timestamps each dispatch in its own pass,
  and never through `device_time`.
- **Everything `just bench` reports** — the prefill wall, the decode step, the
  session, the sweep, the cross-engine columns. Those are wall clocks around
  whole processes.
- **Everything in `attention`, `norm`, `dense`, `sconv` and `argmax`.** The
  double division was six expressions in one module; `norm` carries its own copy
  of the arrangement and divides once.

### What the instruction the block is made of issues at

Every "of peak" figure this file has recorded about the matmul divides by a
bandwidth. **That denominator says nothing about this kernel.** The block moves
2.35 GB in 52 ms, which is 45 GB/s against a bus that streams 725 — six percent —
so what bounds it is on the far side of the arithmetic, and the arithmetic has a
ceiling of its own nobody here had measured.

Four ways of asking this part for it, all in registers, no load and no barrier
inside the loop, 640 threadgroups of 256, warm and best of five:

    what issues                            device            rate
    the matrix instruction, four held     16.56ms    25.3 TFLOP/s
    the same chain held once               4.76ms    22.0 TFLOP/s
    the same four on 16-bit operands      15.87ms    26.4 TFLOP/s
    scalar fused multiply-adds             4.56ms    23.0 TFLOP/s

**This part has one arithmetic rate.** The matrix instruction is not a datapath
of its own — it is within a tenth of the scalar one — so a kernel carrying 512
multiply-adds an instruction has the same ceiling as one carrying four. Two
things fall out of that and both were on somebody's list:

- **16-bit operands buy four percent of issue rate**, not a doubling. Apple's
  guidance for this hardware is most emphatic about them and mlx's quantised
  kernels take it completely, and on this part the instruction does not go
  faster for it.
- **A serial accumulator costs thirteen percent.** The brief's leading
  hypothesis was that the inner loop is serial per fragment and wants software
  pipelining; four independent accumulators against one is 25.3 against 22.0, so
  the whole prize for breaking that dependency is 1.15× and the block already
  holds four.

Against that ceiling, the shipped block, at the shapes its tables are read
across — the same sitting as the four rows above, so that what divides and what
is divided were measured on one clock:

    a prefill of              device     achieved   of the instruction
     2048  q_proj, tiled     3.87ms   17.8 TFLOP/s      70%
     2048  a routed bank    17.00ms   12.1 TFLOP/s      48%
     4096  q_proj, tiled     7.63ms   18.0 TFLOP/s      71%
     4096  a routed bank    26.31ms   15.7 TFLOP/s      62%
     8192  q_proj, tiled    15.16ms   18.1 TFLOP/s      72%
     8192  a routed bank    52.41ms   15.7 TFLOP/s      62%

**What is left is a third to a half, not the factor of four the roofline
suggested.**

### Where the block's time goes, now that the clock is right

The same limiter arms as before, since the ratios were never wrong, plus the one
term no ablation across the bus can reach. At 8192 tokens, warm, swept both ways
and agreeing to a hundredth of a millisecond at every arm:

    without                                       q_proj, tiled   a routed bank
    nothing — the kernel                         15.16ms  100%    52.42ms  100%
    the weight it reads                          14.70ms   97%    50.93ms   97%
    the input rows it walks                      14.68ms   97%    50.91ms   97%
    the table it decodes through                 14.80ms   98%    51.32ms   98%
    three of every four fragment loads           14.18ms   94%    49.35ms   94%
    three quarters of the multiply-accumulates    7.87ms   52%    30.88ms   59%
    the two barriers a step                      13.50ms   89%    47.38ms   90%

**The barriers are ten percent and they are the largest thing here that is not
the instruction.** A step is one group of codes wide, so a reduction of 4096 is
128 steps and **256 barriers a pass**, every one of which stops all eight
simdgroups until the slowest arrives. The arm removes them and races — same
loads, same decode, same multiply-accumulates, none of the waiting — so what it
prints is an upper bound and not an estimate.

**And the accounting closes to about a fifth.** Cutting three of four
multiply-accumulates takes 41%, which by arithmetic puts the instruction near 55%
of the kernel if the term is linear in its count; the barriers are 10%, the
fragment loads 6%, and the three byte-count terms 8% between them where they do
not overlap. The two thirds that used to be unattributed is now a fifth, and the
fifth is the staging stores, the address arithmetic and the loop.

### What mlx's quantised matmul costs at these same shapes

**Nobody here had ever dispatched it.** Every cross-engine reading in this file
is a whole prefill or mlx's dense kernels; the closer question is what its
*quantised* MoE matmul costs at the two shapes this file's tables are read
across, and `just qmm-probe` is what asks it. MXFP4 at group 32, four bits,
float32 — this checkpoint's own format and this engine's own dtype — one dispatch
each, best of three:

                              ours       mlx    behind   mlx, nothing quantised
     2048  q_proj, tiled     3.87ms    3.51ms    1.10×                   3.28ms
     4096  q_proj, tiled     7.63ms    6.86ms    1.11×                   6.41ms
     8192  q_proj, tiled    15.16ms   13.29ms    1.14×                  12.27ms
     2048  a routed bank    17.00ms   10.70ms    1.59×                        —
     4096  a routed bank    26.31ms   21.07ms    1.25×                        —
     8192  a routed bank    52.41ms   41.47ms    1.26×                        —

**The existence proof the brief was written from does not exist for this
problem.** 63.5% of peak is a dense GEMM figure, and mlx's dense fp32 GEMM here
is 22.4 TFLOP/s — 89% of the ceiling above, which is the strongest corroboration
that ceiling has. Its *quantised* kernels run 19.3 to 20.7 TFLOP/s, which is 76
to 82%. This engine runs 48 to 72%. **The gap is real and it is 1.1 to 1.3× at
every length but the shortest, not 2.6×**, and at 8192 it is the 1.28× the
end-to-end session shows.

**The row that is worse than the rest is the routed bank at 2048**, at 1.59×.
That is the shape where this arrangement is weakest rather than where the kernel
is: 12288 rows over 256 experts is 48 rows an expert, and 32 does not divide 48,
so **one block in three straddles an expert boundary** and every straddle runs
the whole reduction twice — once per expert its rows name, with the rows that are
not that pass's zeroed. mlx cuts the same block at the run boundaries and
computes each side once. At 8192 the runs are 192 rows, 32 divides that, no block
straddles anything, and the row closes to 1.26×.

One column in that table is not about this engine at all. mlx reaches
`gather_qmm_rhs` — one expert per row of a plain `[M, K]` input — only when the
indices are declared sorted; unsorted, the same call takes 127.75 ms at 8192
against 41.47. **Three times, for exactly the arrangement `ExpertGrouping`
already pays for on this side.**

### What `gather_qmm` does that this kernel does not

`fp_quantized.h` over `steel/gemm/mma.h`, and the metallib in `reference/.venv`
at mlx 0.32.0 carries it at this checkpoint's format:
`mxfp4_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2`.

**Its shape is smaller and its inner loop is the same one.** 16 rows by 32
columns over 1 × 2 simdgroups — 64 threads where this runs 256 — with `BK` of 32,
which is this kernel's step; and its `k` loop is barrier, fill both tiles,
barrier, multiply `BK / 8` times, barrier for barrier what this does. `TM` and
`TN` are both 2, so **a simdgroup there holds four accumulators against four
fragment loads, which is 1:1** — the ratio a taller block was wanted for here,
run by the kernel that was the reason to want it.

Three things it does differently, and none of them is the shape:

- **It never calls `simdgroup_load`.** `BaseMMAFrag::load` reads the two elements
  each lane owns straight out of threadgroup memory through a static
  lane-to-coordinate map, and takes the transpose by swapping the two strides it
  indexes with. This kernel issues a `simdgroup_load` a fragment and a
  *transposing* one for the weight.
- **It has no answer tile.** `store_result` writes fragments to device memory
  where they lie and `store_result_slice` is how it refuses a ragged edge.
- **It walks runs rather than passes.** A block whose rows name several experts
  is cut at the sorted index list's boundaries and each run gets its own gemm and
  its own slice of the store, where this kernel runs the whole reduction once per
  distinct expert and zeroes the rows that are not that pass's.

**And one of the three claims about mlx this file has carried since A8 is
false.** Its steel GEMM does run 64 to 128 threads and its dense 64×64 tile does
reuse fragments 2.7:1, both read off that metallib. But mlx **does** ship fp32
quantised matmuls — `mxfp4_qmm_t_float_gs_32_b_4` is in the same binary, and all
four of its quantised families are instantiated at `float` beside `float16_t` and
`bfloat16_t`. That claim is what the 16-bit arm was argued from.

### What the generated code says, and what Apple will not let you read

The toolchain is installed off the nix PATH: `xcodebuild -showComponent
MetalToolchain` gives the asset and `Metal.xctoolchain/usr/bin` holds `metal`,
`metallib`, `metal-objdump` and `applegpu-nt`, the AIR-to-native translator.
Three readings come out of it and the one everybody wants does not.

- **The pipeline reports this part's own maximum 1024 threads a threadgroup for
  both production entries**, so nothing about the block's occupancy is decided by
  its register allocation.
- **The AIR the frontend emits keeps every fragment loop rolled**, with `lhs`,
  `rhs` and `sums` in `alloca`s indexed by the loop variable rather than in
  registers. That is not what runs — the unrolling and the allocation are the AGX
  backend's, and a full-unroll pragma on all four loops changes the emitted
  metallib by one line. **So the metallib is not evidence about the inner loop's
  schedule**, which is exactly the reading a source-level intuition would most
  like to take from it.
- **The native object builds and does not disassemble.** `applegpu-nt -arch
  applegpu_g15d` translates it into a Mach-O for this part — `mma_matmul_rows` is
  4736 bytes of `__compute` against `packed_matmul_rows` at 7136 — but the
  toolchain ships no instruction printer for the `agx` targets
  (`metal-objdump: no instruction printer for target agx3---macho`).

**A ceiling measured in registers is what replaced that reading**, and it answers
the question the listing was wanted for — how many of this kernel's cycles are
the instruction — without needing a model of what the part does with what it is
issued.

### The one thing that changed, which is four percent

**The block writes its answer over the staging it has finished with.** The two
are never both live: nothing reads a staged tile after the last
multiply-accumulate of the last pass and nothing writes the answer before it. The
threadgroup's declaration goes from 22688 bytes to 13984, and a barrier one
threadgroup waits at is covered by whatever else the core is holding. **How many
that is, is arithmetic off somebody else's measurement**: `RESIDENCY`'s sweep put
four threadgroups a core at a declaration of 20 KiB, three from just above it to
26.67, and two past that — a core holding 80 KiB, which puts this block at three
before and five after.

One barrier is added, before the first fragment store, because the store lands
where another simdgroup may still be reading.

**Four percent at every shape and every length, answering bit for bit** — which
it must, since what moved is only where a fragment is parked between the last
multiply and the write-out. Paired, alternating, order reversed, seven pairs,
against the commit before it:

    at 8192, --numerics production      before      after   change   ranges
    prefill wall                      17233.9ms  16844.6ms   -2.3%   apart, 7 of 7
    prefill device                    12190.6ms  11794.9ms   -3.2%   apart, 7 of 7

    at 2048
    prefill wall                       5687.9ms   5641.3ms   -0.8%   across, no claim
    prefill device                     3386.3ms   3286.7ms   -2.9%   apart, 7 of 7

### Two doors that closed on arithmetic and are now closed on a measurement

**The taller block fits.** "A block twice as tall does not fit — 36160 bytes
against the 32768 limit" was the reason the height was closed; with the answer
tile aliased it is 18752 and the sweep runs it. At 8192 it is worth **nothing**
(101% on both shapes) and at 2048 it costs **twice** the routed bank, because a
routing's runs do not fill 64 rows at that length. Same door, better reason.

That shape also exposed a hole in its own sweep: the fixture had 74 rows where a
64-row block needs 128 before a call reaches it at all, so it would have answered
the *reference tile's* bits under the swept block's name and passed. It is 142
rows now.

**And 16-bit operands are refused again, this time at equal occupancy.** The arm
now reads its narrow tiles out of a scratch the width the answer wants, so what
it changes is the operand and not how many threadgroups a core holds:

    at                            the block   at 36    at 40    at 44
    2048   q_proj, tiled             3.87ms     +8%      +9%      +5%
    2048   a routed bank, grouped   17.00ms     +9%      +9%      +5%
    8192   q_proj, tiled            15.15ms     +9%      +9%      +5%
    8192   a routed bank, grouped   52.39ms     +8%      +8%      +4%

Drift is flat at 1.5 to 2.1e-4 across reductions 32 to 4096, two decades above the
block's own, which is the operand rounding and not a chain. **Slower and less
accurate**, and the ceiling table above says why it could not have been faster:
the instruction issues at the same rate on both.

### The threadgroup sweep, re-taken on the kernel that ships

At 8192 tokens, warm, both ways, every arm answering the shipped block bit for
bit:

    a threadgroup                reuse     q_proj, tiled     a routed bank
     64 threads, 1×2 simdgroups  2.0:1  348.71ms  2302%   1053.61ms  2008%
    128 threads, 2×2 simdgroups  1.3:1   15.67ms   103%     54.01ms   103%
    256 threads, 2×4 simdgroups  1.0:1   15.15ms   100%     52.46ms   100%
    512 threads, 2×8 simdgroups  0.7:1   25.33ms   167%     85.53ms   163%
    256 threads, 4×2, 64 rows    1.3:1   15.33ms   101%     52.77ms   101%

The shipped width is still the best of them and the reuse column still runs
backwards against the clock. **What mlx's quantised kernel does at 64 threads is
not what this one does at 64 threads**, and the sweep says why: a block's
threadgroup memory is a property of the block, so 64 threads here buys the same
14 KiB with a quarter of the work to hide a load behind. mlx's is 16 by 32 rather
than 32 by 64, so its 64 threads carry a quarter of the memory too.

### What did not move

**No token changed and no default changed.** The flag stays defaulted to
reference, the reference path is byte-identical, and the recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both backends write.
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` and both MMA
floors are where they were, and the acceptance rows on the packed heads are
unmoved.

**A decode step is untouched**, which it has to be: the block is not dispatched
at any shape a decode step has. Paired against the commit before the kernel
change, seven pairs: 19.446 against 19.429 ms, device 18.615 against 18.617,
ranges across, no claim. **The null pair was run and read −0.0%**, ranges across —
`just bench HEAD HEAD decode`, same binary both arms, seven pairs — and the host
was settled to 0.76% on the wall and 0.37% on the device before the sitting.

**K1's kept-cache advantage is untouched.** Both openings, both engines, one
sitting each, three pairs, ours under `--numerics production`:

    opening        ours kept   mlx-vlm kept   behind
    2048            19.49 s       14.35 s     1.36×
    8192            32.75 s       25.49 s     1.28×

    the per-turn bookkeeping inside it
    2048             1.024 ms     31.449 ms
    8192             1.115 ms     66.687 ms

Ours flat in the context where theirs is linear, exactly as K1 measured it.

**Our own prefill wall, unsampled, under the production flag**:

    tokens    wall      device
     2048    5.51 s     3.29 s
     4096    8.98 s     6.06 s
     8192   16.79 s    11.80 s

### The floor this stops at, and its arithmetic

**A stopping condition is a floor you can prove, and this is one.**

The instruction this kernel is made of issues at 25.3 TFLOP/s on this part, and
that is the same rate its scalar fused multiply-add issues at — so no arrangement
of this arithmetic goes faster, whatever it is written in and whatever width its
operands are. The routed bank at 8192 runs 824.6 GFLOP in 52.41 ms, which is 15.7
TFLOP/s, which is **62% of that**.

What the remaining 38% is, priced rather than guessed: barriers 10%, fragment
loads 6%, the three byte-count terms 8% between them, and about a fifth
unattributed. **Removing every one of the priced terms entirely — which no kernel
can, they are the mechanism and not an overhead — would put the call at 39.8 ms
and 82% of the instruction.** mlx's quantised kernel is at 79%. So the whole gap
to the other engine is inside the terms this table has now priced, and there is
at most **1.6× between here and a kernel that is nothing but its instruction.**

**What is worth trying next, and what it would cost:**

- **A step of 64 codes instead of 32**, which halves the barrier count. The
  measured upper bound is the barrier arm's **10%**. It costs the staging
  doubling to 26 KiB, which takes the core back to three threadgroups from five,
  and it moves the scale read inside the staging loop because a step would then
  span two groups. Half a day, and the two effects run opposite ways, so it is a
  sweep rather than a change.
- **Fragment loads through a lane-to-coordinate map instead of
  `simdgroup_load`**, which is what mlx's own quantised kernel does and is the
  one structural difference between the two inner loops. The measured upper bound
  is the fragment-load arm's **6%**, and the transposing load on the weight is the
  one most likely to be paying more than the two element reads it owes. A day,
  and it needs the lane-to-coordinate map verified against this part rather than
  inherited from mlx's header.
- **Both at their ceilings is 16%**, which is 52.41 ms to 44.0 and 74% of the
  instruction. **That closes about two thirds of the distance to mlx and does not
  close it**, so a third thing would still be wanted — and the honest reading of
  the fifth that is unattributed is that it is where the third thing is.

**Both were taken and the 16% was 0.8%** — see "The two levers between 62% and
74%", where the step is built, swept and refused at 107% and the per-lane read
ships at 0.7 to 0.9%. What each of the two prices above gets wrong is the same
step: an arm that *removes* work prices the work, and neither lever removes any.
The barrier arm's 10% is the imbalance the staging leaves rather than the count of
barriers, so halving the count returned a sixteenth of it; the fragment arm's 6%
is three of every four reads of threadgroup memory rather than the instruction
around them, so rewriting the instruction returned a twelfth.

**What this does not license is a claim that the engine is finished.** It is
1.28× behind at the 8192 opening and 1.36× at 2048, and the gap is real. What
this milestone says is that the gap is a third rather than a factor of four, that
the ceiling it is measured against is now a measured number rather than a
bandwidth that never applied, and that the two terms worth taking next are priced
before anyone spends a day on them.

## The four levers the matmul had left

**The three tables below that were taken on the device's clock read four times
too fast, and every ratio in them stands** — the roofline and limiter table, the
threadgroup sweep, and the 16-bit one. See "The clock every table in this crate
was read off" above, which is where the corrected ones are; each arm divided by
the same four, so what this section concluded about each lever is what it
concluded. **The per-kernel table, the prefill wall and the session are not
affected**: those come off a sampled prefill and a wall clock rather than off
`device_time`.

**Every one of them was measured and every one of them is refused, and the
measurement that did it is one cheap question asked first.** This milestone
opened with four candidates for the prefill gap — whether the matmul had become
bandwidth-bound, a block-height refactor, fragment reuse, and the thread width —
and a fifth held back for last, 16-bit operands. The first was taken first
because it could re-rank the rest, and it did: **it re-ranked all of them to
zero.**

**No kernel changed. What this milestone ships is a shape the block can be swept
at, and five tables saying what the sweeps found.**

### Which kernel the closed doors were closed on

**Every "this is not bandwidth-bound" finding in this file was taken on the
reference tile.** A3's 96-fold expert-weight re-read costing 10.4%, A4's whole
limiter table — both of them on the kernel A7 then replaced with one 2.85×
faster. **A kernel that was issue-bound can become
bandwidth-bound when it gets faster**, and at that moment every byte-count lever
those findings retired comes back.

So `what_a_prefills_blocked_matmul_is_bound_by` is A4's instrument pointed at
`mma_matmul_rows` and `mma_matmul_grouped` — the two entries that actually run a
prefill under the production flag — at the three lengths a coding session lives
between.

### The premise this milestone had to correct before it could use it

**The brief this milestone was written from argued that 16384 flatters the
matmul, and that has not been true since A8.** The argument was that attention
is quadratic and the matmul is not, so at 16384 attention was 43 s of the work
and the matmul's share looked small: 64% of the passes against 56%.
`mma_attention` took the two attention rows nineteen times down. The in-model
per-kernel table now reads, under `--numerics production`, one sampled prefill a
row:

    tokens   the grouped matmul   the row-tiled matmul   both   of every pass
     2048           1.88s                1.23s          3.11s        88%
     4096           3.28s                2.46s          5.74s        85%
     8192           6.13s                4.84s         10.97s        83%
    16384          11.81s                9.65s         21.46s        80%

**So no length flatters it any more.** The matmul is where a prefill's time is at
every length, and the reason to measure at 2048 to 8192 is the plainer one: that
is where a coding session opens, and a prefill there is seconds where 16384 is
half a minute.

### Whether the matmul is bandwidth-bound now, which is no

**Two readings, because either alone is misread.** The first is the roofline —
what a call must fetch once each, over the device's own clock, against the 725
GB/s `what_a_streaming_read_achieves_on_this_machine` measures:

                    q_proj, tiled                   a routed bank, grouped
    tokens   taken   must move   at 725 GB/s      taken   must move   at 725 GB/s
     2048   1.01ms    0.08 GB   0.10ms   10%     4.42ms    1.44 GB   1.99ms   45%
     4096   1.98ms    0.14 GB   0.20ms   10%     6.83ms    1.74 GB   2.41ms   35%
     8192   3.95ms    0.28 GB   0.38ms   10%    13.57ms    2.35 GB   3.24ms   24%

**The compulsory floor is a quarter of the grouped call and a tenth of the tiled
one, and it falls as the prompt grows** — the weight is the same 256 experts
however many rows read it, so a longer prefill amortises it further. There is a
factor of four to memory on the row that has least of it.

But a roofline cannot tell a kernel bound by traffic it *must* move from one
bound by traffic it *re-reads*, and those have opposite consequences for the rest
of this list. So the second reading is the limiter table — the shipped kernel
with one term at a time replaced by something that costs an instruction over the
same operands and cannot be folded away. **Warm, and swept both ways**, because
the arms are four and five percent apart and this file has the case on record
where that was a clock: at 8192 tokens, the two passes agreeing to a hundredth of
a millisecond at every arm.

    without                                       q_proj, tiled   a routed bank, grouped
    nothing — the kernel                         3.95ms   100%      13.57ms   100%
    the weight it reads                          3.77ms    96%      13.07ms    96%
    the input rows it walks                      3.78ms    96%      13.04ms    96%
    the table it decodes through                 3.83ms    97%      13.23ms    97%
    three of every four fragment loads           3.70ms    94%      12.86ms    95%
    three quarters of the multiply-accumulates   2.70ms    68%       9.47ms    70%

**Confining the whole weight working set to 2 KB is four percent.** Every byte of
the 2.35 GB the call must fetch, plus all 1791 weight reads it issues where 256
would do — the same bytes decoded, the same arithmetic, a working set that fits
in a cache line — and the kernel comes back 96% of itself. **The matmul is not
bandwidth-bound and it is not close.** A3's 10.4% is now 4%.

**The input is four percent and the decode is three.** A4 measured the decode at
30% of the reference tile and could not remove it; the block amortises it eight
times as far, and what is left of it is 3%. That term is finished.

**The multiply-accumulate is the only thing here with a factor in it.** Cutting
three of every four leaves every fragment load where it was and takes 30 to 32%,
so the instruction is about 41% of the kernel. Nothing in the brief's four levers
points at it.

### What fragment reuse would have bought, which is six percent at its ceiling

**"What the taller block would cost at its floor" recorded the reuse half as
unmeasurable without a refactor, and on this kernel it turns out to be
measurable without one.** That section is about the *attention* block — A8 left
its two multiplies at 1:1 and named the obstacle — and the matmul's block has
the same obstacle and the same ratio. A simdgroup here holds
`MMA_FRAGMENTS_DOWN × MMA_FRAGMENTS_ACROSS` = four accumulators and issues two
`lhs` and two `rhs` loads to feed them, which is **1:1** where mlx-vlm's 64×64
steel tile runs **2.7:1**. A taller block is the way to buy that ratio and it
costs a doubled floor — on the attention kernel that is a measured 2.4 to 3.8× on
a short turn over a long session, which is exactly the shape K1's kept cache
produces.

The arm above buys the ratio's ceiling for nothing instead. Hoisting the two
fragment loads out of the step's `k` loop drives all sixteen multiply-accumulates
off one set of fragments — **4:1, past mlx-vlm's 2.7:1**, with no taller block and
no floor to pay. It answers wrongly, so what it prices is the loads alone and it
is an upper bound on anything reuse could return.

**It is worth five to six percent**, and it is an upper bound rather than an
estimate. So the most a taller block could return on this kernel is that, and the
floor table on the kernel beside it prices what the same trade gives away at 2.4
to 3.8× on a short turn over a long session. **That closes the height on the
matmul, and it closes it without the refactor that was thought to be needed to
ask.**

**And a block twice as tall does not fit on this part at all**, which the sweep
found rather than argued: two staged tiles, an answer tile and the block's expert
list at 64 rows come to 36160 bytes against a threadgroup's 32768, and the
pipeline refuses it.

### What the block's threadgroup is worth, which is nothing this side of 256

**mlx-vlm runs its steel GEMM at 64 to 128 threads where this ships 256, and
nothing here had ever tried another width.** That figure and the 2.7:1 above have
been asserted in this file since A8 and were never checked against anything; they
are checked now, against the metallib this repo's own oracle runs — mlx 0.32.0 in
`reference/.venv`, whose `steel_gemm_fused_*_bm64_bn64_bk16_wm1_wn2` symbols and
`steel/gemm/mma.h` give a 64×64 tile over `wm × wn` simdgroups of 32, which is 64
threads at `1×2` and 128 at `2×2`, and 32 accumulators against 12 fragment loads.
**Both hold.** The reason was not that it looked
unpromising: the width was a host-side constant as well as a source one — the
grid is `rows.div_ceil(rows) × out_dim.div_ceil(cols)` threadgroups of it — so an
entry compiled at one width and dispatched over a grid sized for another leaves
output no threadgroup reached, which is a wrong answer rather than a slow one.
`Block` is what closed that, and it is the one thing this milestone ships.

At 8192 tokens, warm, both ways, every arm answering the shipped block bit for
bit:

    a threadgroup                reuse     q_proj, tiled     a routed bank, grouped
     64 threads, 1×2 simdgroups  2.0:1   68.18ms   1726%    210.06ms   1546%
    128 threads, 2×2 simdgroups  1.3:1    5.37ms    136%     17.82ms    131%
    256 threads, 2×4 simdgroups  1.0:1    3.95ms    100%     13.59ms    100%
    512 threads, 2×8 simdgroups  0.7:1    6.31ms    160%     20.94ms    154%

**The shipped width is the best of the four and the reuse column runs backwards
against the clock.** A narrower threadgroup gives a simdgroup more fragments to
hold, so 64 threads is 2.0:1 where 256 is 1:1 — most of the way to the ratio a
taller block was wanted for — and it is **seventeen times slower**.

The reason is that a block's threadgroup memory is a property of the block and
not of its threads. Two staged tiles and an answer tile come to about 22 KiB
whatever covers them, so a core holding 32 KiB holds one threadgroup either way —
and at 64 threads that is two simdgroups a core against eight, with the same
memory and a quarter of the work to hide a load behind. **This is the second
measurement in this milestone saying that fragment reuse is not what this kernel
wants**, and it is the one that says why.

### What sixteen-bit operands cost and what they give up

**Apple's guidance for this hardware is most emphatic about 16-bit types, and
mlx's own quantised matmul takes it completely — `qmm` ships `bfloat16` and
`float16` variants and no `float32` one.** (Its *dense* steel GEMM does ship
fp32, which is worth saying because this file has asserted otherwise: the
metallib in `reference/.venv` at mlx 0.32.0 carries
`steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm1_wn2`.) A8 declined it and named the reason
to respect: this flag's two sides sum the same exact products in different
orders, and a 16-bit operand rounds the product itself, so it would have to move
into the conditioning table rather than sit beside it.

It is in the conditioning table. The accumulator stays `simdgroup_float8x8` —
the instruction takes that mixed form on this family — so nothing about the
summation order moves and what is priced is the operand alone. Drift from an f64
accumulation of the same products:

    a reduction   the reference   the block   16-bit operands
       32            8.3e-8         1.3e-7        1.8e-4
      128            1.4e-7         3.5e-7        1.7e-4
      512            8.4e-8         6.6e-7        1.5e-4
     2048            1.1e-7         1.5e-6        2.1e-4
     4096            1.0e-7         1.6e-6        1.8e-4

**The shape of that column is the finding rather than its size.** The block's own
drift grows with the reduction, because its fragment accumulator is one running
sum and a longer chain is a worse one. A drift that is *flat* is not a chain at
all — it is the operand, arriving already rounded and staying that far off
however few products are summed. Which is exactly what A8 predicted, and it sits
two decades above the block at every length.

**And it is slower.** Three paddings rather than one, because the shipped stride
is derived against a four-byte staged element and at two bytes 36, 40 and 44 all
put a fragment's eight rows on eight distinct banks and should read alike on the
argument alone. They do not:

    at                            the block   at 36    at 40    at 44
    2048   q_proj, tiled             1.01ms     +6%      +3%      +2%
    2048   a routed bank, grouped    4.42ms     +6%      +2%      +3%
    4096   q_proj, tiled             1.99ms     +6%      +2%      +2%
    4096   a routed bank, grouped    6.83ms     +5%      +1%      +2%
    8192   q_proj, tiled             3.95ms     +6%      +2%      +2%
    8192   a routed bank, grouped   13.58ms     +6%      +2%      +2%

**A single-stride reading would have priced the padding and called it the operand
width** — 36 costs six percent where 40 and 44 cost one to three. At the best of
them it is still slower at every shape. **Refused on both counts**, and both are
asserted, so either turning stops the tier rather than printing quietly.

### A weight whose rows were all the same weight

**The instrument had a hole in it and the first arm this milestone wrote fell
into it.** `Case::seeded` built every synthetic MXFP4 code from
`noise.next() % 16`, and `Noise` is a power-of-two linear congruence whose bit
`i` has period `2^(i+1)`. Four bits off the bottom are the state's 8 to 11, so
the code stream repeated **every 4096** — which is the width of every reduction
in this checkpoint, and therefore exactly one row of a weight.

So every row of every expert of every synthetic weight held the same codes. A
kernel that read some other column's codes, or some other expert's, would have
answered bit for bit what the right one answers; the limiter tables in this file
have been pricing the weight through its *scale* alone, because 11 and 5 are not
powers of two and the scales never had the problem. `no_two_rows_of_a_synthetic
_weight_hold_the_same_codes` is what says it cannot come back, asked at the
reduction this checkpoint has and either side of it.

**The tables were re-taken after the fix and did not move**, which is what the
limiter arms above are: the weight arm still reads 96% with the codes genuinely
differing per column.

### What did not move, which is every kernel and every token

**No kernel changed and no default changed.** `Block::SHIPPED` emits the prelude
the module constants emitted, byte for byte; every swept shape answers the
shipped one bit for bit, because changing the threadgroup changes which
simdgroup owns an output element and not the order it is accumulated in; and the
flag stays defaulted to reference.

**All 695 cases pass against a real checkpoint** — 644 in the gated tier in 654 s
and the 51 of the timing tier in 1427 s, both to completion without
`--no-fail-fast`, run against the tree these commits leave. The recorded continuation `[656, 13, 623, 180069, 86333,
60500, 220, 23]` is what both backends write,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` and both shape
floors are where they were, and the acceptance rows on the packed heads are
unmoved. The tier is five cases larger than it was: the block's limiter table,
the 16-bit table, the threadgroup sweep, and two ordinary cases holding the
fixture and the swept shapes to what they promise.

**The in-model per-kernel table was re-taken after the refactor and did not
move**, which is the check that matters because the refactor is on the shipped
dispatch path — `entry`, `rows_a_read` and `blocks` all read the block off the
compiled entry now:

    tokens    the grouped matmul       the row-tiled matmul
              before     after         before     after
     2048      1.88s     1.90s          1.23s     1.24s
     4096      3.28s     3.28s          2.46s     2.45s
     8192      6.13s     6.13s          4.84s     4.84s
    16384     11.81s    11.81s          9.65s     9.63s

**And the session is where it was.** Both openings, both engines, one sitting
each, three pairs, ours under `--numerics production`:

    opening        ours kept   mlx-vlm kept   behind    K1's behind
    2048            19.68 s      14.38 s      1.37×        1.40×
    8192            36.16 s      27.24 s      1.33×        1.32×

    the prefill inside it, which is the turn's time to first token
    2048             6.34 s       3.69 s      1.72×
    8192            18.76 s      13.62 s      1.38×

**K1's kept-cache advantage is untouched**, which is the one architectural
advantage this engine has measured: the per-turn bookkeeping is **1.013 ms at
the 2048 opening and 1.267 at 8192** against the reference's 31.44 and 66.26 —
ours flat in the context where theirs is linear, because a mark is four windows a
layer and a count of keys where a snapshot copies the whole KV.

**The null pair was run and it read −0.3%, ranges across, no claim** —
`just bench HEAD HEAD decode`, same binary both arms, seven pairs. The host was
settled to 0.33% before the sitting and read 19.386 to 19.450 ms, against 19.406
to 19.468 before any of this was written.

**Our own prefill wall, unsampled, under the production flag**, at the three
lengths — ours alone, since the cross-engine column above is the one taken in a
sitting with the other engine in it:

    tokens    wall      device
     2048    5.74 s     3.39 s
     4096    9.33 s     6.26 s
     8192   17.29 s    12.19 s

### What this milestone leaves

**The matmul's remaining time is the matrix instruction and whatever is around
it, and nothing on the brief's list reaches either.** Cutting three of four
multiply-accumulates takes 31%, which by arithmetic puts the instruction near
41% of the kernel if the term is linear in its count — labelled arithmetic
because that is what it is. Everything else here is worth three to six percent
apiece.

**What the arms do not do is sum, and they are not asserted to.** Each is a
marginal removal against the whole kernel and several of them overlap: the
fragment-load arm and the accumulate arm reach the same registers, and the input
arm and the weight arm reach the same cache. So what is left over is not a
number this milestone has — what it has is that no byte-count term is worth more
than six percent, and that the largest single term is an instruction.

**That is the measurement the next milestone wants**, and it is a different
instrument from this one: the terms left are inside the threadgroup — the staging
stores, the two barriers a step, the loop — rather than across the bus, and an
ablation that confines a device read cannot separate them.

**What is now cheap that was not.** The block's shape is a value, so a height, a
width, a column span or a staging step is one arm of a sweep rather than a
milestone. The threadgroup sweep here cost an afternoon and the four levers it
retired had cost four milestones of deferral between them.

**What this does not license is a claim that the engine is finished.** It is 1.33×
behind at the opening a coding turn has, and the gap is real; what this milestone
says is that none of the five things anyone had written down would close it, and
that the two doors A3 and A4 closed on the reference tile stay closed on the
kernel that replaced it.

## What a conversation costs when its cache is kept

**Every figure in this file below this line is one request**, and the workload
this engine exists for is not one request. A coding session sends the same
context back turn after turn with a little added each time; the server built a
fresh cache for each of them, so a conversation of a few thousand tokens paid for
all of them on every turn. Nothing here had ever measured that, because a
measurement of one call has no *between* for a kept cache to live in.

`just bench-session` is that measurement. An arm is a session: 8192 tokens
opened with, then five turns that each add 256 tokens of question, carry the
previous reply back, and decode 64 tokens of their own — so the prompt grows
8192, 8512, 8832, 9152, 9472, which is the shape a coding turn has.
`inkling_core::workload::Session` owns it, beside the rest of what this repo
measures over.

**The opening is the workload's number rather than the cheapest one to measure.**
A coding turn opens nearer 8192 than 2048, and what a kept cache is worth moves
with the length — so both are reported below and the shorter one understates it.

**Three pairs each, one sitting each, the order flipped each pair**, on this
repo's default checkpoint under the default numerics. The arm is a number rather
than an executable — `--reuse-tokens 0` is the server as it was, the default
bound is the server as it is — which is the shape `bench-numerics` puts one word
through.

    turn   prompt   prefilled  ─── not kept ───   ──── kept ────   change
                    not/kept     wall     first    wall    first    wall
      0      8192   8192/8192  48.98 s   46.77 s 49.07 s  46.86 s   +0.2%
      1      8512   8512/ 321  50.81 s   49.01 s  5.76 s   3.95 s  -88.7%
      2      8832   8832/ 321  52.84 s   51.04 s  5.86 s   4.06 s  -88.9%
      3      9152   9152/ 321  54.88 s   53.07 s  5.78 s   3.98 s  -89.5%
      4      9472   9472/ 321  57.00 s   55.19 s  5.90 s   4.08 s  -89.7%
    session                   264.51 s           72.36 s          -72.6%

**The session is 264.51 s against 72.36 s**, three pairs of three moving the same
way with the ranges apart. What a turn prefills stops growing: 321 tokens every
turn after the first, which is the reply carried back, the question added, and
the one token a generation has to be handed. By turn four that is 9472 tokens of
prefill against 321, and the wait for the first token is 55.19 s against 4.08 s.

**Turn zero is the sitting's own control and it reads +0.2% with the ranges
across, no claim.** Nothing is kept yet, so both arms do identical work — and
what the harness reads when there is nothing to read is what says the rest of the
column is not the harness.

**The same sitting at a 2048-token opening reads -61.6%**: 85.10 s against 32.68,
turn four 20.44 s against 4.87, and turn zero -0.2% with the ranges across.
**So the shorter session understates this by eleven points**, and it understates
it for the reason the whole thing exists: what a kept cache saves is the
re-prefill, which grows with the conversation, while what it cannot save — the
opening prefill and the decode steps — does not grow with it nearly as fast. A
figure taken at 2048 is a floor rather than an estimate.

### What it costs

**The keys were already held between requests and this did not change that.** A
layer's span lives on the device inside the wrapped weights, grows by doubling
and is never given back; a fresh cache sets its count to zero and leaves the
buffer where it is. So a kept conversation adds no KV at all on the device path.
What it does add, held between turns, is the ids — eight bytes a position, 256
KiB at the 32768-position bound — and the `ModelCache`, whose key vectors are
empty on the device path and whose four convolution windows a layer are **4.92
MiB across the stack**. On `--backend cpu` the keys *are* those vectors, and the
bound is the whole of what stands between a server and a resident set that only
climbs: 10752 MiB at 32768 by the KV table below. Both figures are arithmetic
from the config's widths, and the second is that table's.

**Inside a request the mark is 9.84 MiB** — the same four windows a layer read
back off the device and cloned on this side — taken before the generation and
dropped after it. Peak RSS on a single-request run is unmoved because nothing on
that path takes one; a served request's peak gains that and nothing else.

**And the arrangement's own bad case is 1.81 ms a turn.** That is the matching,
the mark, the resume and the recording, on the miss path — where it also builds
the fresh cache the server allocated on every request before any of this — with
1.11 ms on the hit path. Against a turn that is seconds either way, a miss costs
what a miss should cost. **It is also flat in the context**: 1.10 and 1.52 ms at
a 2048-token opening against 1.11 and 1.81 at 8192, over conversations four times
the length. A mark is four windows a layer and a count of keys, and neither of
those grows with the keys.

### The reference does this too, and it is not a gap we opened

**mlx-vlm ships Automatic Prefix Caching**, and the brief for this work said to
find out. For a plain attention model it is block-level and pageable; for one
whose cache is not block-concatenable it falls to an exact path — a whole
prompt-cache snapshot taken at a prefix boundary, matched by the ids that
produced it. Inkling is that second kind, because every layer carries four
short-convolution windows beside its keys, so the reference reaches the same
arrangement this repo reached and from the same constraint. Two things about it
are worth saying precisely: it is **off unless asked for** — `APC_ENABLED`
defaults to `0` and the manager is wired through mlx-vlm's server rather than
through `generate_step` — and its cache is **not trimmable**, since `ArraysCache`
declares no `trim` and so `CacheList.is_trimmable()` is false here. The reply
cannot be taken back out of a kept cache; the snapshot stands in for that.

`reference/scripts/bench_session` is that machinery behind the same
`name value unit` contract, so the same three-pair alternating sitting can be run
against it. **Its own two arms are 81.79 s against 25.70 s, -68.6%** at the 8192
opening and 32.54 against 14.57, -55.2%, at 2048 — the same effect, growing with
the length the same way ours does, three of three with the ranges apart. So this
is not a differentiator and saying otherwise would be the substantial lie.

**Both engines keeping, ours under `--numerics production`** — which is the
arithmetic the cross-engine column is quoted under:

    opening      ours     mlx-vlm   behind
       2048   20.09 s     14.57 s   1.40x
       8192   33.73 s     25.51 s   1.32x

three of three with the ranges apart at both lengths. Both prefill the same 320
tokens a turn; what is left is prefill throughput and the decode step, which are
the two rows the sections below are about. **The gap narrows as the session
lengthens** — 1.40× to 1.32× — because more of what is left is decode, where this
engine is the faster of the two.

**Where the two do differ is what keeping costs, and that difference grows.**
Per turn:

    opening      ours     mlx-vlm    ratio
       2048   1.10 ms    31.98 ms      29x
       8192   1.11 ms    67.09 ms      60x

**Ours is flat in the context and the reference's is linear in it**, which is
exactly what the two designs predict: a mark is four windows a layer and a count
of keys, and a snapshot is a copy of the whole KV. Neither number matters against
a turn that is seconds; what the row says is that only one of the two would still
be small at 32768.

**One caveat on the reference's column**: its not-kept session ranges 76.55 to
92.16 s across the three pairs at 8192, and 27.35 to 42.72 at 2048, where ours
ranges 264.32 to 264.69 and 85.00 to 85.20. The mean is reported as measured and
the spread is the reference's own.

### What it is not

It is not a prefix-cache service and does not pretend to be one. **A layer holds
one span and one window pair on the device**, so two conversations interleaved
through it would overwrite each other's keys — there is no arrangement of this
that keeps both. One entry, and the entry is a conversation; a prompt that does
not extend it replaces it.

The matching is all of the kept ids or none. One position is kept and it is the
end of what was recorded, so a prompt that agrees for a while and then parts
company has nothing here to start from. That covers a coding turn, which is an
exact extension of the turn before it, and nothing wider.

**And it changes no token.**
`a_session_served_from_a_kept_cache_produces_the_tokens_a_cold_one_produces` runs
four turns warm and four turns cold against the real checkpoint and compares the
ids of every token of every turn. Ids and not text: a reply that reads the same
is a reply that might have moved a token. That is what makes this a latency
optimisation rather than an approximation, the same thing
`speculation_changes_no_token` says about the other one.

**Capping the windowed layers is still not done and is still the larger prize** —
see the KV table below, where 35 of 42 layers retain 4.6× what their window could
ever read at the 8192 tokens this session opens at. Nothing here forecloses it:
the mark carries a window's rows and a count of keys, and a ring-addressed span
would change what the count means without changing that a mark is those two
things.

### What did not move

**No kernel was touched and no numerics decision was made.** The flag stays
defaulted to reference. All 690 cases pass against a real checkpoint — 642 in the
gated tier and the 48 of the timing tier, which **runs to completion without
`--no-fail-fast`** — and the recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is still what both backends
write, `--backend cpu` included. The tiers are eighteen cases larger than they
were and every one of the eighteen is new: the mark against the synthetic stack
and against a run of layers on the device, the matching and the bound against
their own cases, the session's shape, and the cold-against-warm session on real
weights.

The two floors,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert`, the
acceptance rows on the packed heads, the resident sets the gated tier bounds and
the three decode-step tables are what those 48 assert, and all 48 passed. **No
kernel here could have moved them**: nothing in this milestone is inside a
forward pass. The host says the same — a decode step read 19.42, 19.44 and 19.44
ms in the three settling runs taken before the first sitting and 19.39 to 19.46
in the three before the last, against 19.36 to 19.44 over the five taken before
any of it was written — all of it inside that settling's own 0.46% spread.

**The null pair was run and it read +0.1%, ranges across, no claim** —
`just bench HEAD HEAD decode`, same binary both arms, seven pairs. That control
is what says the harness is not inventing the column above, and turn zero of the
session sitting says the same thing inside the sitting itself.

## Against the reference, end to end

**This is A2's sitting and it has been re-taken** — see "One clean cross-engine
sitting", which is seven pairs at `42effa1` with all twenty-four readings claims
where this one had two that were not. **The reference's four wall times agree
across the three milestones to 0.5%** and ours are what A5 and A8 left, so what
this section has right is everything about the shape and what it has stale is
this engine's own column. It is kept as taken because the diagnosis under it was
made against these rows.

**This file had no defensible headline figure and that was a real gap.** Prefill
was last measured against the reference two milestones ago; the decode figure
quoted against it dated from before the run was pipelined, while our own number
moved five times underneath it; and nothing here had ever measured the one number
a user actually feels — a prompt and its answer, prefill and decode weighed
against each other rather than quoted apart. `just bench-engines` is that
measurement, and it is the first in this file whose other arm is not this engine:
an arm is an executable that prints `name value unit` lines, so
`reference/scripts/bench_engines` is mlx-vlm behind the same protocol,
alternating with ours pair by pair.

**Seven pairs, one sitting, the order flipped each pair**, on
`models/Inkling-Small-mxfp4-mtp4` — the packed heads, which is where this
engine's own best figure is and is not this repo's default checkpoint. Twenty-two
of the twenty-four readings moved the same way in all seven pairs with the ranges
apart; the two that did not are named where they fall. Both engines were given
the same prompt tiled to the same length and asked for the same number of tokens,
and **the two produced the same tokens**:
`[3004, 49159, 13, 200010, 200001, 200008, 976, 1825]` at 97,
`[2454, 402, 1617, 2316, 2543, 306, 9707, 290]` at 385 and
`[3665, 478, 25, 478, 117867, 382, 391, 6120]` at 769, ours and mlx-vlm's alike.
That is what makes this a comparison of two engines rather than of two workloads.

**What a user waits, prompt and answer together:**

    prompt × generated    ours k = 0   ours k = 2    mlx-vlm    k = 2 against it
     97 × 128                3.551 s      2.723 s    3.209 s          1.18× ahead
    385 × 128                7.584 s      6.377 s    3.673 s          0.58×
    769 × 128               10.803 s      9.487 s    4.186 s          0.44×
     97 × 512               15.531 s      9.703 s   12.131 s          1.25× ahead

**So this engine wins the short prompt and loses the long one, and the crossover
is a number rather than a direction.** At a 97-token prompt, speculating two
deep, we start 299 ms behind at the first token and take 6.19 ms less per token
after it — so the wall times cross at **about 49 generated tokens**, and both
rows at that prompt are past it. At `k = 0` there is no crossover at any prompt
length, and at 385 and 769 tokens there is none at any depth: our decode step at
those contexts is already slower than the reference's, so every token generated
widens the gap rather than closing it. **That is the finding this table was taken
for**, and it is not the one the milestone expected.

**The prefill, which is the first token and everything under it:**

    tokens        ours    mlx-vlm     gap    ours at k = 2   the reference's stack
       97       561 ms     283 ms   ×1.98          582 ms                  260 ms
      385      2732 ms     707 ms   ×3.87         2786 ms                  680 ms
      769      5396 ms    1171 ms   ×4.61         5457 ms                 1140 ms

**The `k = 2` column is the control and it is meant to be dull**: a prompt takes
one token out of its last position however many positions it had, so speculation
cannot reach a prefill, and the three pairs agree to within 3.7%.

**The reference did not move and the 97-token row did.** `just prefill-bench`
reads 0.26, 0.68 and 1.14 s for the transformer stack on its own, the same three
figures this file has quoted since P4, and the last column is there to say that
the comparison was never unfair in mlx-vlm's favour: its full `[1, L, vocab]`
projection and its argmax are 23 to 31 ms on top, so time-to-first-token and
stack-only are the same measurement to within 3%. What changed is ours — 1.22 s
to 0.56 s at 97 tokens — and it changed because the device argmax took the host
round trip out. The ×4.7, ×4.7 and ×4.8 this file carried is now **×2.0, ×3.9 and
×4.6**.

**A decode step, and the column this file has never printed — what it costs at a
context somebody might have:**

    context      ours k = 0   ours k = 2    mlx-vlm      ours k = 2, tokens/s
     97              23.54        16.86      23.04        59.3 against 43.4
    385              38.21        28.28      23.36        35.4 against 42.8
    769              42.58        31.73      23.74        31.5 against 42.1
     97 → 609        29.30        17.87      23.19        55.9 against 43.1

**The reference's decode step is flat in the context and ours is not**, and that
is the whole of why the two long-prompt rows read as they do. 23.04 to 23.74 ms
across an eight-fold context is mlx-vlm holding still; 23.54 to 42.58 is this
engine walking the span. **The 20.59 ms this file quotes for a decode step is
taken at an eight-token context**, and it is a true figure about a context nobody
has.

**Both halves of that paragraph were then measured past 769 tokens and both are
wrong out there.** See "Where a decode step goes as the context grows" below: the
reference is flat to 769 and takes a threefold step at 2048, and this engine's
own row is no longer the one that grows fastest. The table above is a paired
sitting inside a range where a coding workload has not begun, and it is kept for
what it is.

**The one row that is not a claim is the honest one to name.** `97 × 128` at
`k = 0` reads 23.54 ms against 23.04 with the ranges across each other and six of
seven pairs agreeing — no claim by this file's own standard. So at a short
context and without speculation **the two engines' decode steps are
indistinguishable**, and every other decode row in the table is a claim. The
other is that pair's wall time, which inherits it.

**The mean is not the median at the two long prompts, and what separates them is
one step.** Ours reads 38.21 ms a token at a 385-token context against a median
of 32.62, and 42.58 against 36.44 at 769 — and the longest step of each
generation is **step 1** at 736 ms and 783 ms, where at 97 tokens the longest is
25.55 ms and falls at step 125. That is not a decode step: it is what the prefill
deferred, arriving on the step after it, and a 769-token prefill retains 13741
MiB over 1278 buffers that are released when its command buffer completes. The
reference has no such step — its own maxima are 27.7, 33.3 and 35.1 ms against
medians of 22.9, 23.1 and 23.6. So our steady-state step is 32.6 and 36.4 ms and
the 0.7 to 0.8 s belongs to the prefill row above rather than to this one. It is
in the wall time either way, which is why the end-to-end table is the one to read.

**What this says about where the work is**, taken apart row by row rather than
asserted. At 769 tokens in and 128 out we are 5.30 s behind at `k = 2`, and that
divides exactly: **4.29 s of prefill and 1.01 s of decode**. At 97 in and 512 out
we are 2.43 s ahead, and that divides too — **2.71 s won on the decode step
against 0.29 s given away at the prefill**. And the same pair at `k = 0` is 3.40 s
behind, of which 3.13 s is the decode step and 0.27 s the prefill: without
speculation the context growth alone loses that row.

**So prefill is not the whole remaining gap.** It is the whole of it only where
the prompt is long against its answer; the moment a generation is long, the decode
step's growth with the context is the larger number, and at `k = 0` it is the only
number. Both are the same fact from two sides — a prefill reads the model once per
token, and a decode step walks every key it has.

## Where a decode step goes as the context grows

**Every decode figure above, either engine's, was taken at a prompt of 769 tokens
or fewer.** A coding turn opens at thousands and grows all session, so "the
reference is flat in the context" was a claim about eightfold, made where the
workload has not begun. Swept out to where one lives — ours by
`what_a_decode_step_costs_as_the_context_grows`, the reference by
`reference/scripts/context_sweep.py`, one sitting each, unspeculated:

    context      before   ours   mlx-vlm    ours tokens/s   ours peak   mlx-vlm peak
       97        21.99  19.99     23.58     50.0 v 42.4      0.23 GiB    130.99 GiB
      385        31.04  21.34     23.67     46.9 v 42.3      1.24 GiB    131.94 GiB
      769        36.03  21.91     24.52     45.7 v 40.8      1.26 GiB    132.97 GiB
     2048        41.56  24.85     77.85     40.2 v 12.8      1.98 GiB    135.86 GiB
     4096        50.19  26.09     74.93     38.3 v 13.3      2.78 GiB    136.70 GiB
     8192        67.17  28.65     78.70     34.9 v 12.7      4.35 GiB    138.60 GiB
    16384          —      —       79.36        —  v 12.6         —       142.44 GiB
    32768          —      —       91.27        —  v 11.0         —       150.18 GiB

**The reference is not flat, and that overturns the premise this milestone
started from.** 23.6 to 24.5 ms from 97 to 769 is the whole of the range the
cross-engine table covers; at 2048 it is 77.9 ms and it stays near 78 to 16384.
Whatever produces that step is not diagnosed here — the reference is the target
rather than the object of study — but the shape is not a slope, it is a
discontinuity between 769 and 2048 with a plateau after it. **So flat to 769 was
never an existence proof of flat at 8k**, and the thing it was being used to
prove is one this engine now does better than the proof.

**The reference's five long rows do not reproduce and this paragraph is
withdrawn** — see "The reference's decode step, which is the figure that did not
survive". Re-run on the same script, the same pinned mlx-vlm and the same default
contexts, it reads 26.33 at 2048 and 43.01 at 32768 against the 77.85 and 91.27
above, with the peak column identical to the digit at all eight rows. **There is
no step and there is no plateau**; the reference is a shallow slope the whole way
and always was.

**This engine is ahead at every context measured**, where before the split it
lost 385 and 769 and won the rest. It is 2.7× ahead at 8192 and its own row grew
by 8.7 ms across an 84-fold context where it used to grow by 45. Both columns are
one sitting apiece and neither is paired — what makes them readable is that the
effects are 2× to 6× and this host drifts 1.7%. The eight-token figures under
"Sampling on the device" are what is paired, and they are what moved least.

**The 2.7× is withdrawn with the row it divides by, and the figure is 1.01×.**
Our own column reproduces to a tenth of a millisecond at every context, so what
this cost is the reference's arm and nothing about this engine — and the lead
that is left runs 1.17× at 97 keys to 0.98× at 32768. **This is the fourth figure
in this file not to survive being questioned and the first whose other half was
another engine.** The sentence above about the effects being "2× to 6×" against a
1.7% drift is exactly the reasoning that let it stand unpaired, and the effect it
was protecting was not the engine's.

**Our column has since moved and "What a decode step waits on" above is where.**
It reads 18.16 ms at 97 keys and 27.20 at 8192 against the 19.99 and 28.65 here,
which is 9.2% off the shortest row and 5.1% off the longest — a lane holding four
chunks of its weight row and an untiled dispatch given a threadgroup of its own.
**The slope is the one thing that got slightly steeper**: 1.12 µs a token of
context against the 1.07 below, because what came off was the part of a step that
does not grow with the context and the part that does is where it was. The
reference's column here was not re-taken; the cross-engine table in that section
is the side-by-side one, and it covers 97 to 769.

**Linear, and the slope is the number to carry.** Ours is 19.99 ms at 97 keys
and 28.65 at 8192, which is **1.07 µs a token of context** where before it was
5.6 — and `what_the_attention_step_costs_as_the_context_grows` says why it stays
linear rather than turning: the 7 global layers cost 0.076 µs a key and hold that
figure to 65536, where the 35 windowed ones are flat past their window. Carried
out, that is about 55 ms a token at 32768 and 130 at 100k, against a reference
measured at 91 ms at 32768. **Those two are arithmetic and are labelled as such**;
what is measured stops at 8192, and what stopped it is that a prefill to 16384
costs five minutes here and to 32768 fourteen.

**Both ends of that comparison have since been measured and both were wrong.**
Ours reads 33.92 and 43.74 ms of device time at 16384 and 32768 against the 55 ms
the extrapolation put at 32768, so the line over-predicts; the reference reads
43.01 at 32768 against the 91 quoted here. The extrapolation being 26% high and
the figure it was held against being 2.1× high happened to cancel into the right
sign, which is the argument for measuring the two rather than carrying either.

**That last sentence is wrong and "Where a prefill's time goes as the prompt
grows" below is where it is measured.** A prefill of 16384 tokens costs 133.63
seconds on this engine and 32768 is about six minutes by arithmetic. Nothing
between that reading and this one changed the prefill path; what the two figures
are is a measurement and an estimate that was never taken.

### The keys a sequence keeps, which the window does not bound

**35 of the 42 layers can never read past their own last 512 keys and all 42
retain every key the sequence has seen.** `KeyValues::reserve` allocates against
the keys a sequence has, in powers of two, and consults
`AttentionConfig::sliding` nowhere. What
`what_a_context_costs_in_keys_and_values` weighs is that against the window:

    context      the spans   windowed    if capped   over
       97           42 MiB     35 MiB       42 MiB   ×1.0
      769          336 MiB    280 MiB      196 MiB   ×1.7
     8192         2688 MiB   2240 MiB      588 MiB   ×4.6
    32768        10752 MiB   8960 MiB     1932 MiB   ×5.6
    65536        21504 MiB  17920 MiB     3724 MiB   ×5.8

**The architecture note below claims 28 KiB/token and a 1M-token context under
30 GiB, and that is a claim about a design this engine does not implement.** It
is arrived at by growing the 7 global layers and holding the 35 windowed ones at
their window; here all 42 grow, and in float32 rather than the note's bfloat16.
The third column is this engine's own power-of-two rule applied to the keys a
layer may reach rather than the keys it has seen — the case asserts that a global
layer's two columns agree, which is what makes it that rather than a second guess.

**It is not a regression and it is not the reference's fault either**:
`InklingModel.make_cache` hands every one of mlx-vlm's 42 layers a plain
`KVCache` too, and its own peak grows 131.0 to 150.2 GiB across the sweep against
our 0.23 to 4.35. **Nothing here was changed.** Capping a windowed layer means a
ring buffer — the kernel's key addressing, `rewind`'s interaction with a
speculative round, and what a prefill writes — and the size of the prize is the
column above rather than a paragraph.

## Where a prefill's time goes as the prompt grows

**The milestone this section was taken for began from a figure that is not
true of this engine.** The decode sweep above closes by saying that what stopped
it at 8192 is "that a prefill to 16384 costs five minutes here and to 32768
fourteen", and a prefill of 16384 tokens at that commit costs **133.63 seconds**.
Measured one length at a run by `bench prefill`, warm, against
`models/Inkling-Small-mxfp4`:

    tokens        ours    tok/s     mlx-vlm    tok/s      gap    mlx-vlm peak
      769       5.40 s      142     1.171 s      657    ×4.61        —
     2048      12.66 s    161.7      2.66 s    769.7    ×4.76    135.6 GiB
     4096      25.36 s    161.5      5.61 s    730.7    ×4.52    141.2 GiB
     8192      54.25 s    151.0     13.05 s    627.9    ×4.16    155.7 GiB
    16384     133.63 s    122.6     34.31 s    477.6    ×3.89    200.8 GiB

**Our column is a milestone behind and the one that supersedes it is under
"What the two turns are worth"**: 11.32, 22.56, 46.77 and 109.51 s at the four
lengths, which is 180.9 to 149.6 tokens a second and ×4.26 to ×3.19. It is kept
here because everything the rest of this section diagnoses was diagnosed against
it, and a table whose rows moved under the diagnosis that used them would be two
sittings spliced.

**Both columns have since been taken in one sitting and the reference's is
unchanged** — 2.66, 5.62, 13.05 and 34.37 s against the 2.66, 5.61, 13.05 and
34.31 above, which is the arm four milestones declined to re-measure being right
to re-measure once. **The production path at 16384 is 33.12 s against that
34.37**; see "What a prefill costs against the reference, both numerics, one
sitting", which is where the four lengths are read under both words.

The 769 row is the cross-engine table's, kept for continuity; the four below it
are new. Ours is time to first token and the reference's is its transformer
stack alone, which is the same measurement to within the millisecond our own
head and argmax cost at these lengths. Each column is one sitting a length and neither is
paired — the effects are decades and this host drifts 1.7%, which is the same
standard the decode sweep above is taken to and no better.

**Prefill is very nearly linear, and 122.6 tokens a second at 16384 is not the
shape "getting worse with length" describes.** 2048 to 4096 is ×2.003 across a
doubling. What curvature there is arrives at the top — ×2.14 then ×2.46 — and
where it comes from is one row of the table below. **The reference falls
faster over the same range**: 769.7 to 477.6 tokens a second against our 161.7 to
122.6, so the ×4.76 at 2048 is ×3.89 at 16384. That is a comparison and not an
explanation; nothing here diagnoses mlx-vlm.

### Which of a prefill's kernels grow with the prompt and which grow faster

`where_a_long_prefill_spends_its_time` divides one sampled prefill a length up
by kernel, with **the 35 windowed layers charged apart from the 7 global ones** —
`Kernel::under` is what makes that a row rather than an argument, and it splits
them in every table this repo takes rather than only here. What the device timed,
the four lengths beside each other:

    kernel                     2048      4096      8192     16384    8192→16384
    global attention        647.80ms    2.35 s    8.75 s   43.74 s         ×5.00
    packed_matmul_grouped     4.78 s    9.54 s   18.87 s   38.15 s         ×2.02
    packed_matmul_rows        4.18 s    8.51 s   16.93 s   34.16 s         ×2.02
    windowed attention        1.55 s    3.34 s    6.88 s   13.99 s         ×2.03
    short_conv               49.64ms  225.39ms  481.66ms  900.39ms         ×1.87
    dense_matmul             89.48ms  183.74ms  374.38ms  756.67ms         ×2.02
    group_by_expert          67.48ms  139.25ms  282.31ms  567.00ms         ×2.01
    swiglu                   49.96ms  100.33ms  201.40ms  403.15ms         ×2.00
    moe_combine              17.15ms   34.67ms   69.50ms  139.30ms         ×2.00
    every pass                11.50 s   24.46 s   52.91 s  132.97 s         ×2.51
    the command buffers       10.67 s   22.64 s   49.30 s  125.73 s         ×2.55

**The 16384 column is the one the occupancy turn moved**, and what it reads now
is under "What the two turns are worth": 31.99 s of global attention, 33.19 and
26.50 of the two matmul rows, 11.44 of windowed attention, and 106.63 s of
passes. The shape of the table is what this section is about and the shape did
not change — exactly one term is still superlinear, and it is still the global
row.

The last two rows are the two clocks and the gap between them is the pass span's
own over-reporting, which is 7.24 s at 16384 and is what a boundary a dispatch
costs. **Everything divided below divides by the passes**, because the rows are
passes; the wall times in the table above it are what the wall claims are made
in.

**Exactly one term is superlinear.** Every other row lands between ×1.87 and
×2.03 when the prompt doubles, where the global row is ×5.00. The growth column
is the last doubling rather than a fit over all four, because the 2048 column is
the first prefill this process ran and its short rows carry that: `short_conv`
reads 49.64 ms there against 225.39 at twice the prompt, and `rms_norm` 65.59
against 30.65. The rows that dominate do not, and the two matmul rows are 54% of
the passes at 16384.

Taken apart the other way: everything that is not attention costs **4.54, 4.58,
4.55 and 4.59 ms a token** across the four lengths, and the 35 windowed layers
cost **21.6, 23.3, 24.0 and 24.4 µs a token each**. So a prefill is about 5.4 ms
a token whatever the prompt, plus a global term that is not.

That term is quadratic and a little worse than quadratic in the measured range:
per token squared it reads 154, 140, 130 and **163 nanoseconds**, so the last
doubling costs ×5.00 where a square costs ×4. Per token it is 45.2, 81.9, 152.6
and 381.4 µs against a windowed layer's flat 24.

**`what_a_prefills_attention_costs_as_the_prompt_grows` says the same thing off
one dispatch**, at n query rows over n keys with no model around it, and the two
agree to within 1% at every length — 95.21 ms against 95.4 for a windowed layer
at 4096, 6.25 s against 6.25 for a global one at 16384. That is worth having,
because the standalone sweep is thirty seconds where the table above is 3.5
minutes of device time.

### Whether the 6× was there

**It was not, and the windowed bound is why.** The hypothesis this milestone
opened with was that a prefill walks full spans on all 42 layers where 35 of them
should stop at 512 keys — `42 × n²/2` where `35 × n × 512 + 7 × n²/2` is the
work. That is **×4.6 at 16384** and six times in the limit where the linear term
stops mattering, and it was the second of those the milestone was sized against.
A1 had left prefill untouched and flagged the bound as a decode-time fix.

But the bound is in the kernel and not in the split: `reach` is computed from
each query row's own position, so it holds for a call of one query row and a call
of 16384 of them alike. What that costs, priced rather than reasoned about: the
35 windowed layers are 13.99 s at 16384 and the 7 global ones are 43.74, so a
windowed layer costs `13.99/35 ÷ 43.74/7` — **one sixteenth** of a global one at
the same prompt. Walking full spans they would each cost what a global one does,
so the 35 of them would be 218.7 s of passes where they are 13.99: 337.7 s of
passes against 132.97, which is **×2.54** and puts the prefill at about 339
seconds rather than 133.63. Those last two are arithmetic and are labelled as
such — nothing here ran a prefill with the bound taken off.

### What the bandwidth column says now that it divides by the right number

**The declared byte count was wrong at prefill shape and the row that mattered
most read `2 GB/s` because of it.** An attention dispatch charged its keys and
values once — `2 × kv_heads × keys × head_dim` — where a call of `n` query rows
walks them once a row and each of the 32 query heads walks its KV head's span for
itself. Corrected to the reads the dispatch issues, which is the contract a bank
binding 256 experts and reading six already states:

    kernel                    2048     4096     8192    16384
    global attention      744 GB/s 819 GB/s 880 GB/s 704 GB/s
    windowed attention    697 GB/s 696 GB/s 698 GB/s 698 GB/s

**Both are at this machine and have been all along.** 819 GB/s is the part's
stated figure, and a windowed layer sits at 85% of it at all four lengths. The
global row reads 744, 819, 880 and 704, and the 880 is a rate this part does not
have: either some of those reads are served without reaching memory — 32 query
rows next to each other walk almost the same keys — or the declared figure is
still counting reads the walk does not make. **Which of the two is not decided
here**, and it does not have to be for the row to be read: at every length the
kernel is within a factor of 1.2 of the machine, where a kernel with arithmetic
to do would be decades off it.

**So the global row is not a slow kernel, it is a kernel reading the same keys
32768 times.** At 16384 tokens one global layer issues 4.4 TB of key and value
reads against a distinct span of 134 MB, and does the 2.2 TFLOP underneath them
at 352 GFLOP/s — decades under this machine's arithmetic and 86% of its memory.
What makes it faster is reading less.

**The two matmul rows' 360 and 277 GB/s are not read the same way**, for the
reason "Why the two tiled rows report bandwidths a factor of two apart" gives at
length: `PackedBank::moves` charges a whole weight per tile, so those are
amplification factors and the distinct bytes are decades below. Nothing about
them moved here.

### Whether a long prefill reads one expert's weight more than it must

**It reads it 96 times where once would do, and that is worth 10%.** The two
matmul rows are 72.3 s of a 132.97 s prefill at 16384 tokens — larger than both
attention rows together — and the arithmetic that says they should not be is
plain: a routed bank runs six rows a token over 256 experts, so an expert is
named by `6n/256` rows — 2.3 at a 97-token prompt, 18 at 769 and **384 at
16384** — where `ROWS_A_TILE` is 4. Every four of those 384 rows walk that
expert's whole 4.5 MB weight for themselves.

**The count was never in doubt and the price was.** A tile reads
`out_dim × in_dim / 4` codes per row of output whatever the run, so the bytes a
row is *charged* are flat by construction; what is not flat is how many of them
have to come from memory. `how_often_a_long_prefill_reads_one_experts_weight`
holds one 1.07 GB bank of 256 experts fixed and dispatches it at four run
lengths, sorted, so the only thing varying is how many tiles want the same
weight — with `group_by_expert` dispatched apart and taken off, since a pass
over the rows would otherwise put a linear term in the column the question is
about:

    rows an expert       rows     reads      a call       a row    declared
    4                    1024        1×       2.8ms      2758ns     1141 MB
    24                   6144        6×      18.0ms      2928ns     6845 MB
    96                  24576       24×      73.8ms      3001ns    27380 MB
    384                 98304       96×     299.4ms      3046ns   109522 MB

**The first row is the ideal rather than a baseline to beat**: four rows an
expert is one tile an expert, so that arm reads each weight exactly the once it
must. 96-fold re-reading costs **10.4% of the time a row** against it — 2758 ns
to 3046 — over a declared figure that grows 96-fold beside it. So all but a
tenth of those reads are served without reaching memory: 109522 MB declared in
299.4 ms is 366 GB/s, of which 3.8 GB/s is distinct.

**So it is real and it is a tenth rather than an order.** 10.4% of the two
matmul rows at 16384 is about 7.5 s of a 133 s prefill, against the global
attention row's 43.74 s, and 7.5 s is a *ceiling* rather than an offer: the only
lever that reaches it is a taller tile, and that has now been swept three times —
on a bank small enough to cache, on one too big, and here — turning at four,
at three, and refused above three by the speculative round `a_calls_rows_share_a
_weight_read_only_where_they_name_one_expert` pins out of the tiled path. The
premise that the grouping reads an expert's weight 96 times where once would do
is exactly right; the inference that it is where a long prefill's time went is
not.

### What is left, sized and not taken

A threadgroup that carried `R` query rows through one tile of keys would divide
the global row's reads by `R`, and the per-row arithmetic need not move at all:
tiles outer and rows inner keeps each row's walk over the same tiles in the same
order, which is what `the_bounded_loop_is_the_unbounded_one_bit_for_bit` rests
on. What has to go somewhere is each row's running peak, total and weighted sum
for the length of the walk, and the staging already holds 19 KB of the 32 an
Apple GPU allows. **Three of every four reads are redundant before any of that**:
four query heads share one KV head under this checkpoint's grouping and each of
the four walks that one span for itself.

**That is the next milestone and it is not this one.** What this one found is
that the defect it was called to fix is not present, that the one superlinear
term is inherent to full attention rather than a missing bound, and that the
term is 33% of a 16384-token prefill against the two matmul rows' 54%.

**It was taken, it is bit-identical, and it is slower — see the two sections
below.** The `R` is real and the reads do divide by it; what the paragraph above
gets wrong is `704 GB/s`, which is a rate over reads a walk *issues* and not
over traffic it makes.

**Carried out, 32768 tokens is about 5.9 minutes, and that figure is
arithmetic.** Split the 133.63 s wall at 16384 by the passes' own shares — which
assumes the over-reporting falls evenly across the kernels, and is a splice of
two runs: 43.96 s is the global row and 89.67 s is everything else, which is
163.8 ns a token squared and 5.47 ms a token. At 32768 those are 175.9 s and
179.3 s, so **355 seconds**. It is not fourteen minutes. The quadratic constant taken is the worst
of the four lengths, and nothing past 16384 was run: a paired sitting of
32768-token prefills is hours and the sweep above already answers which term
grows.

### Carrying a block of query rows through one tile of keys

**The lever the section above sized was taken, and it is bit-identical to the
kernel it replaces and slower than it at every height.** `QUERIES_A_BLOCK` gave
a threadgroup a block of consecutive query rows of one head instead of a single
row: the block staged one tile of values, read each key of that tile once, and
the `R` dots the key feeds came off that one read.

**It was then given back, and what is in the tree is the kernel that was always
there.** The block lives in `c2095cb` with its own bit-for-bit case and its own
sweep, and both are re-runnable from that commit; what is *not* worth carrying
is the block itself, which cost 0.4% of a prefill's device clock at a height of
one for a lever the section below shows could never have paid. The reading that
kills it needs no block at all — it is a two-line mutation of the kernel that
ships — so the disproof stays free and permanent while the disproved thing does
not stay at all.

**Tiles outer and rows inner, which is what makes it identical rather than
close.** A row takes part in exactly the tiles its own `[reach, last)` gave it,
in the same order, with its own `held` rather than the block's — so its running
peak, total and weighted sum meet the same values in the same order they met a
threadgroup at a time. A row with no live key in a tile is skipped rather than
rescaled by `exp(peak - peak)`, which is a NaN where its peak is still
`-INFINITY`. What a *block* reads is the union of its rows' walks rather than
the sum, and `keys_a_call_walks` is that: a global prefill's `n²/2` becomes
`n²/8 + n/2` at four rows, ×3.99 at 2048 tokens.

**`a_block_of_query_rows_is_a_row_at_a_time_bit_for_bit`, in `c2095cb`, is the
case, and it is on the bits rather than on the floats** — `-0.0` and `0.0`
compare equal as floats and are two different answers. Sixteen cases at heights
of two and four
against a height of one, each driven unsplit and through an eight-way fold:
six with a short last block, eight windowed — six of them holding a block whose
rows do reach back to different keys, which is the thing the block is easiest to
get wrong and the thing a windowed case does not automatically ask — and a
prompt from nothing whose first block's rows are one, two, three and four keys
long. **Not one bit of one element moved**, at either height.

**And the sweep is emphatic in the wrong direction.** One dispatch of `n` query
rows over `n` keys, both kinds of layer, against the same dispatch a row a
threadgroup — `what_a_prefills_attention_costs_at_each_height_a_block_carries`,
in `c2095cb`:

    tokens   a block      global   against  window 512   against   the stack
      2048         1     94.04ms     ×1.00     45.18ms     ×1.00       2.24s
      2048         2     98.74ms     ×0.95     47.84ms     ×0.94       2.37s
      2048         4    130.15ms     ×0.72     63.60ms     ×0.71       3.14s
      2048         8   does not compile at this height
      8192         1       1.26s     ×1.00    199.83ms     ×1.00      15.84s
      8192         2       1.28s     ×0.98    211.25ms     ×0.95      16.39s
      8192         4       1.65s     ×0.77    280.67ms     ×0.71      21.35s
      8192         8   does not compile at this height

**It generalised to windowed layers and that is not the good news it sounds
like**: the same tiling, the same bound, the same losses — ×0.71 at four rows on
both kinds of layer at both lengths. Eight rows does not compile, the arrays a
block carries being 3 KB a row against the 32 KB an Apple GPU allows. The widest
threadgroup the pipeline reports is the device's own 1024 at every height, which
is the one place this side could have seen register pressure, and it saw none.

### Whether a prefill's attention is waiting on the keys it reads

**It is not, and that is why the block loses.** `704 GB/s` is
`PackedBank::moves`'s mistake met on the other kernel: it is the reads a walk
*issues* divided by device time, and 32 query heads and their neighbouring rows
walk almost the same keys at almost the same time, so a tile fetched for one is
in cache for the rest. A rate over issued reads says nothing about traffic.

Settled by a mutation rather than by an argument —
`whether_a_prefills_attention_is_waiting_on_the_keys_it_reads` runs the same
kernel with every key and every value read from slot zero. It walks the same
tiles, scores the same keys, takes the same barriers and does the same
arithmetic; its whole working set is one 16 KB tile that never leaves the cache.
What separates the two is the memory and nothing else:

    tokens         layer    the span    one slot     of it
      2048        global     92.67ms     78.06ms       84%
      2048    window 512     44.43ms     37.89ms       85%
      4096        global    335.02ms    276.60ms       83%
      4096    window 512     95.23ms     81.15ms       85%
      8192        global       1.25s       1.01s       81%
      8192    window 512    196.64ms    167.69ms       85%

**So the memory is 15 to 19% of this kernel and the other four fifths are not
the keys.** A lever that removed *every* key and value fetch would take a fifth
off the row; one that divides the reads by four takes at most three quarters of
that, so **11 to 14% at best**. Against it, a block of four costs 39% — 94.04 ms
to 130.15 at 2048 tokens, which is the ×0.72 above read from the slower side.
Both percentages are of the kernel a row at a time, so they subtract. That bound
is generous in the mutant's favour, too: reading one slot also makes the address
loop-invariant, so what it removes is a little more than the memory.

**Which closes M9's hypothesis, deferred five times and now measured rather than
sized.** "Every `(head, query)` threadgroup re-reads all keys; sharing a K/V tile
across queries is the next order of magnitude" is true about the reads and false
about the time. It is the same shape of finding as
"Why the two tiled rows report bandwidths a factor of two apart" — a column that
looked like the largest deficit in the table was an artefact of its denominator
— met on the one kernel that had not yet had its denominator checked.

**What the block cost by existing was 0.4% of a prefill's device time**, and
that is why it is not here: over four alternating pairs at 2048 tokens, every
pair moving the same way and the ranges apart, the device's own clock read
10671.7 ms against 10710.7 with the wall at 12629.6 against 12697.5 and its
ranges across. At a height of one the kernel walked the same tiles under the
same bound over the same grid with the same splits, so the 40 ms bought nothing
but a sweep — and the mutation above reproduces the finding on the kernel that
ships, to the same six percentages. Given back, the two attention rows read
92.67 and 44.43 ms at 2048 tokens against the 94.04 and 45.18 the block's own
height of one measured, which is the 0.4% arriving where it was spent. A decode
step never moved either way: 19.463 ms against 19.475 over seven pairs, ranges
across, three of the seven falling the other way.

### What a prefill's attention is bound by, one term at a time

**The slot-zero mutation said what the memory is worth and nothing about the
other four fifths, and three milestones in a row have now picked a lever from a
number whose denominator was wrong.** So the instrument generalises rather than
the finding. `what_a_prefills_attention_is_bound_by` compiles the shipped source
nine times, each with exactly one term replaced by something that costs an
instruction over the same operands and cannot be folded away, and prices all ten
kernels on the same four dispatches. Every arm answers wrongly and the case
asserts that it does — an arm that still answered what the kernel answers would
be the kernel measured twice under another name.

    without                     2048 global   2048 window   8192 global   8192 window
    nothing — the kernel            92.64ms       44.45ms         1.25s      196.63ms
    the keys and values        78.07ms  84%  37.89ms  85%  1.01s  81%  167.69ms  85%
    the band it derives        71.77ms  77%  32.07ms  72%  1.14s  91%  141.97ms  72%
    the exp's precision        93.28ms 101%  44.74ms 101%  1.26s 101%  198.01ms 101%
    the exp                    93.17ms 101%  44.69ms 101%  1.26s 101%  197.79ms 101%
    the barriers               89.27ms  96%  42.61ms  96%  1.20s  96%  187.85ms  96%
    the tile's two reductions  69.82ms  75%  34.11ms  77%  890ms  71%  150.63ms  77%
    the weighting              80.07ms  86%  38.83ms  87%  1.05s  84%  171.97ms  87%
    three quarters of the dot  82.30ms  89%  39.83ms  90%  1.08s  86%  176.14ms  90%
    the simd_sum               92.26ms 100%  44.24ms 100%  1.25s 100%  195.75ms 100%

**The shares do not sum to one and are not meant to.** Removing a term removes
the instructions that issue it, the registers it held and whatever it was
waiting on, so two terms waiting on each other each read as the whole of the
wait. What the table ranks is which terms are worth anything, and the two that
are worth nothing are as much of the finding as the two that are.

**The transcendental is free and so is the cross-lane reduction, and both were
candidates.** `precise::exp` against `fast::exp2` — the hardware instruction the
reference's softmax is built on — is 101%, and against a two-instruction stand-in
that is not an exponential at all it is 101% again. The `simd_sum` behind every
key's dot is 100%. **Neither is a rounding error inside a large number; they are
nothing**, twice, on both kinds of layer at both lengths, and the milestone that
went to remove either would have found what these two rows say.

**The barriers are 4%.** Four a tile on a threadgroup that is one query row, and
taking all of them out — which is a race and answers accordingly — buys 4% at
every length on both kinds of layer. That figure is generous in the arm's favour:
without the barriers the compiler may also drop threadgroup loads it can no
longer prove another thread wrote.

**The largest single term is a reduction nobody needed to compute twice.** A tile
lands 32 scores in threadgroup memory and then every one of the 256 threads walks
all 32 of them for the maximum and all 32 again for the sum — 64 threadgroup
reads and 64 operations a thread a tile, for two scalars. Cut to one entry apiece
the kernel is **71 to 77%** of itself, and it is the same 23 to 29% on every
column. Nothing about that term is memory, arithmetic the answer needs, or
synchronisation: it is issue slots spent because the running peak and total are
every thread's rather than one thread's and broadcast — which is a trade this
file made deliberately at 32 entries and against a barrier, and which the table
now prices.

**The band this kernel derives is the second, and it is the one term that is
shaped like the reference comparison.** `banded_entry` is `d_rel` device reads and
`d_rel` multiplies made by lane 0 while the other 31 lanes of its simdgroup wait,
once for every key scored — and it is **28% of a windowed layer** at both lengths
against **23% and 9%** of a global one. The split is the band's own extent: a
windowed layer's live keys are all within 512 of the query and every one of them
is inside the 1024-distance band, where a global layer at 8192 reads mostly keys
further back than that and takes the early return. **So the mask this engine does
not materialise is not free** — it is a quarter of the 14.14 s windowed row and a
tenth of the 44.06 s global one, which is about 5.5 s of a 16384-token prefill.

**The weighting and the dot are 13 to 16% and 10 to 14%, and the second of those
carries three quarters of the key traffic with it.** Read beside the slot-zero
arm's 15 to 19% they say the same thing from two sides: what this kernel spends
on the arithmetic an attention step is actually for is a minority of it.

**What is left over is the shape of the answer.** No term here is half the
kernel; the largest is 29% and five of the nine are under 15%. A kernel bound by
one thing has one large row and this has six small ones — which is the signature
of instruction issue rather than of any single resource, and is consistent with
A2's 352 GFLOP/s on a part that does far more: the walk issues about a hundred
lane-instructions per key of which a handful are the multiply-adds the answer
needs.

### What a threadgroup's memory is worth, which is the occupancy term

**Occupancy was on the candidate list and it is turnable by a knob that moves
nothing else.** A threadgroup here is one query row of one head and declares 19
KiB — `staged` is 16 of it — against the 32 KiB a threadgroup may declare on this
part. `how_many_threadgroups_of_a_prefills_attention_a_core_holds` adds a
threadgroup array nobody reads for anything and sweeps its size. The array is
declared last, so every address the kernel uses is where it was; the fill touches
the same 256 floats at every size, so the work is where it was; and the arm is
checked against `staticThreadgroupMemoryLength` rather than hoped for, so an array
the compiler dropped would show as a row that did not move.

**The values are also read where they lie in the lower half of the table**, which
is the only way to get this walk under 16 KiB at all — the staging is what the
memory *is*. It is not a proposal; it is the second half of the same knob.

    the values      a threadgroup   2048 global  2048 window   8192 global  8192 window
    staged                 19 KiB       92.63ms      44.45ms         1.25s     196.69ms
    staged                 23 KiB      120.20ms      57.70ms         1.63s     255.89ms
    staged                 31 KiB      176.35ms      84.34ms         2.41s     373.76ms
    where they lie          3 KiB      114.64ms      52.79ms         1.61s     236.61ms
    where they lie          5 KiB      114.50ms      52.78ms         1.61s     236.05ms
    where they lie          7 KiB      114.29ms      52.69ms         1.61s     235.84ms
    where they lie          9 KiB      114.69ms      53.16ms         1.61s     239.41ms
    where they lie         11 KiB      114.38ms      52.53ms         1.60s     234.90ms
    where they lie         13 KiB       71.12ms      34.33ms      918.95ms     151.72ms
    where they lie         15 KiB       76.72ms      36.73ms         1.03s     163.03ms
    where they lie         17 KiB       92.16ms      43.64ms         1.25s     193.52ms
    where they lie         19 KiB       92.12ms      43.67ms         1.25s     193.54ms
    where they lie         23 KiB      120.52ms      57.62ms         1.64s     256.19ms
    where they lie         31 KiB      177.27ms      84.54ms         2.45s     375.65ms

**The row is a function of the declared memory and of nothing else in it.**
Staged at 19 KiB is 92.63 ms and unstaged at 19 KiB is 92.12, which is 0.6% apart
on a kernel whose values come from threadgroup memory in one and from device
memory in the other — so **at prefill shape the staging buys nothing at all, and
the whole of what it does to this row is declare 16 KiB.** That is not what it was
put there for: the figure the staging was fitted against is one query over 1200
keys, 726 µs to 458, which is a decode-shaped call where a tile is fetched once
and used once. Here 32 heads and their neighbouring rows walk almost the same
keys at almost the same time and the cache is already doing it.

**The turn is at 13 KiB and it is worth 23 to 27%.** 71.12 ms against the shipped
kernel's 92.63 at 2048 tokens — 23.2% — and 918.95 ms against 1.25 s at 8192,
which is 26.5%; the two windowed columns are 22.8 and 22.9%, so the one column
that is worth more than the rest is the one the prefill table is largest on.
Above the turn the row rises the way falling
residency reads — 17 and 19 KiB are one figure, 23 is 30% worse, 31 is 92% worse.
**Below it the row is a flat plateau at 114 ms**, 3 KiB to 11 KiB, five arms
agreeing to 0.4%, which is what a residency capped by something other than memory
looks like — and that plateau is 61% *worse* than the turn rather than better.

**So the curve is a U and this file cannot say why the left arm of it is there.**
More threadgroups a core is better down to some count and worse below it, and the
mechanism — whether it is the core's cache being shared among more streams, or a
scheduler limit the plateau is pinned at — is not settled here. What is measured
is the share: **occupancy is worth 23 to 27% of this kernel, the shipped one is on
the wrong side of the turn, and the only thing standing between it and the turn is
16 KiB of staging that has been shown to buy nothing at this shape.** Nothing was
changed.

### What a prefill's two matmul rows are bound by, one term at a time

**The same instrument on the kernel beside it.** `packed_matmul_grouped` and
`packed_matmul_rows` are 72.3 s of a 132.97 s prefill at 16384 tokens, larger
than both attention rows together, and A3 established one thing about them: that
96-fold re-reading of an expert's weight costs 10.4% against reading it once. So
all but a tenth of those reads are served without reaching memory — and what the
other nine tenths of the row *are* was never asked.
`what_a_prefills_packed_matmul_is_bound_by` asks it the same way, over a
4096-token prompt's `q_proj` on the tiled entry and its routed bank on the
grouped one.

    without                          q_proj, tiled   a routed bank, grouped
    nothing — the kernel                   5.93ms                   19.52ms
    the weight it reads               3.78ms   64%          12.76ms      65%
    the input rows it walks           4.50ms   76%          14.47ms      74%
    the table it decodes through      4.17ms   70%          13.77ms      71%
    three quarters of the columns     3.28ms   55%          11.41ms      58%
    three quarters of the rows        2.49ms   42%           8.41ms      43%
    the scale it walks to             5.99ms  101%          19.50ms     100%
    the simd_sum                      5.96ms  101%          19.54ms     100%

**The two entries answer the same, term for term, and that is the first
finding.** Not one row differs between the columns by more than three points,
over calls whose weights are 8 MB and 1.07 GB and whose experts are 1 and 256.
So whatever separates 38.15 s from 34.16 s in the profile is the number of calls
and their shapes, and not anything either entry does differently — which is what
"Why the two tiled rows report bandwidths a factor of two apart" said from the
denominator's side and this says from the kernel's.

**The weight is a third of it, which is three times what A3 priced.** Pointing
every column of every tile at expert zero's first row — the same bytes decoded,
the same arithmetic, a 2 KB working set — is 64 and 65%. A3's 10.4% is the
*re-reading*, and this is the whole fetch: so about a third of these two rows is
the weight arriving, of which under a third is redundant and the rest is the
model being read. **That part is not a defect and nothing removes it.**

**The dequantisation table is 30%, and it is a memory access nobody counts.**
Every packed byte is two gathers into `ELEMENTS`, a 16-entry constant array
indexed by a nibble, and replacing both with an integer-to-float conversion is 70
and 71% on both entries. `PackedBank::moves` charges the codes and says nothing
about the table they index — so a third of what looks like arithmetic in this
kernel is a dependent load whose latency the byte load has to be waited for
first.

**The input rows are a quarter**, which is the term the column tile was taken
for and which that change did not finish. Confining the two input loads to eight
floats — the same two loads issued from the same place, landing in cache — is 76
and 74%. This file has called the input "small and warm"; it is small, and a
quarter of the row says it is not warm enough.

**The multiply-adds are the largest term either way.** Cutting three quarters of
the columns leaves every weight byte and every input float where they were and
takes only the accumulate — 55 and 58% of the kernel, so the accumulate across a
tile is **45 and 42%**. Cutting three quarters of the rows takes the input reads
with it — 42 and 43%, so **58 and 57%** — and what separates the two arms is 0.79
ms and 3.00 ms, against the 1.43 ms and 5.05 ms the input term costs measured on
its own. **Those two do not reconcile and are not asserted to**: one arm removes
three quarters of the input loads and the other removes none of them and confines
where they land.

**The scale and the cross-lane reduction are nothing**, both entries, the same
way the transcendental and the `simd_sum` are nothing in the attention table.
Four scale loads and sixteen multiply-adds per four bytes read as 100 and 101%,
and one `simd_sum` per output element at the end of a walk thousands of bytes
long reads the same.

**Two of the candidates have no row here because the kernel has no such term.**
There is no transcendental in a packed multiply — the group scale is an exponent
shifted into place and every product is exact — and there is no
`threadgroup_barrier` anywhere in the tile, which is what makes a simdgroup per
tile the unit rather than a threadgroup. So the transcendental's share and
synchronisation's share on both matmul rows are **zero by construction** rather
than by measurement, where on the attention rows they are measured at nothing and
4%.

**All of these tables are on the command buffer's own clock and not on a
per-dispatch sample.** `crate::testing::device_time` divides
`GPUEndTime - GPUStartTime` by the dispatches a buffer holds, so none of them
carries the 18% of over-reporting a compute pass a dispatch adds — see "The
instrumentation is off by default". The counter sample buffers were used for one
thing only: the in-model per-kernel table this milestone re-took as its arbiter.

### What a threadgroup's memory is worth on a kernel that declares none

**A tile of this kernel holds no threadgroup memory at all**, so it runs at
whatever residency this part gives a kernel that asks for nothing — and the
attention sweep above found a row that was a quarter faster on the other side of
a turn. `how_many_threadgroups_of_a_prefills_packed_matmul_a_core_holds` adds the
same dead array, filled by one store a lane at every size so the work does not
move with the knob, and this can only lower residency and never raise it.

    a threadgroup    q_proj, tiled   a routed bank, grouped
     0 KiB                  5.90ms                  19.53ms
     1 KiB                  5.98ms                  19.72ms
     4 KiB                  5.97ms                  19.72ms
     8 KiB                  5.61ms                  18.80ms
    12 KiB                  5.27ms                  16.99ms
    16 KiB                  5.22ms                  16.67ms
    24 KiB                  5.19ms                  16.38ms

**Monotone, reproducible to a hundredth of a millisecond over two passes, and in
the direction that says this kernel runs too many threadgroups a core.** 5.19 ms
against 5.90 is 12% and 16.38 against 19.53 is 16%, bought with memory nobody
reads. Nothing about the walk changed: the same bytes, the same decode, the same
accumulators, one extra store a lane.

**So both dominant kernels are on the same side of the same turn, and neither
was fitted against it.** 23 to 27% of the attention rows and 12 to 16% of the
matmul rows, which over the 16384-token table is about 14 s of the 58.2 s
attention pair and about 10 s of the 72.3 s matmul pair — **24 s of a 133 s
prefill, by arithmetic off two sweeps rather than by a run.** What it would cost
to get is a different question from what these measure: the attention turn is on
the other side of a staging this file has now shown buys nothing at prefill shape,
and the matmul turn was reached here with dead memory rather than with anything a
kernel would want. **Nothing was changed.**

### What the reference's attention kernel does that ours does not

**mlx-vlm's attention is readable Metal in the installed package and this is the
first milestone to read it.** `steel/attn/kernels/steel_attention.h` and the
`mma.h` beside it are what `mx.fast.scaled_dot_product_attention` compiles, and
`mlx_vlm.models.inkling.language` reaches them through a materialised
`[B, H, LQ, S]` additive mask that `banded_additive_mask` builds in a
`mx.fast.metal_kernel` of its own. **This is for understanding and there is a
hard constraint on acting on it — see the section after this one.**

**Four structural differences, in the order they matter.**

**A threadgroup is a block of `BQ` query rows and the multiply is a hardware
matrix instruction.** Ours gives a threadgroup one query row and computes each
score as a lane-strided dot with a `simd_sum` behind it — scalar multiply-adds,
one channel at a time. Theirs holds `Q`, `K` and `V` blocks in threadgroup memory
and drives them through `simdgroup_matrix` 8×8 fragments: `tile_matmad` for the
scores and `MMAFrag_acc_t::mma` for the weighted values, so one instruction
carries 512 multiply-adds where ours carries one. Everything below is small
beside this.

**The softmax lives in registers and takes no barrier.** Ours writes a tile's 32
scores into threadgroup memory, barriers, has every one of 256 threads reduce all
32 for the maximum, barriers, exponentiates, barriers, has every thread reduce
all 32 again for the sum, and barriers. Theirs keeps `max_score` and `sum_score`
as a per-thread register array and reduces a fragment's row with two
`simd_shuffle_xor` steps — lanes 1 and 8 — and no threadgroup traffic at all.
That is the term the table above measures at 23 to 29% of our kernel.

**The exponential is `fast::exp2` with the base folded into the scale.** They
multiply `scale` by `M_LOG2E_F` once, before the loop, so every rescale is a
single hardware instruction; ours is `precise::exp`, deliberately, because every
weight this kernel hands a value comes out of one. **The table above says that
choice costs us nothing** — `fast::exp2` in our kernel reads 101% — so this is a
difference that is not a deficit.

**The mask is a tensor rather than a derivation, and the trade runs our way on
memory and against us on time.** They read `mask[row, col]` and add it; we
compute `banded_entry` on lane 0 of a simdgroup while the other 31 wait. Ours
allocates nothing where theirs allocates 32 heads by the span squared — 34 GB of
float32 at 16384 tokens, which is most of why the reference's peak is 200.8 GiB
there against our 4.35. **And ours costs 28% of a windowed layer and 9 to 23% of
a global one**, by the table above.

**And with a mask given, their loop walks the whole rectangle.** `do_causal` and
`has_mask` are separate function constants and the causal bound is only taken
under the first, so a call carrying an additive mask runs `kb_lim = NK` — every
key for every query, on all 42 layers, with the band's `-1e30` doing the masking.
Ours bounds `[reach, last)` on both ends. **So the reference does about twice the
attention arithmetic we do on a global layer and about thirty times it on a
windowed one, and is still decades faster**, which is what the number below says
and is the whole shape of this comparison.

**One dispatch at our own shapes**, by `just sdpa-probe` — `[1, 32, n, 128]` over
8 KV heads, float32 because that is what our kernel is in, against our own
`what_a_prefills_attention_costs_as_the_prompt_grows` global column:

    tokens        ours   mlx, causal   against   mlx, an additive mask   against
      2048     92.62ms        2.49ms       ×37                  4.95ms       ×19
      4096    334.95ms        8.47ms       ×40                 18.79ms       ×18
      8192       1.25s       31.43ms       ×40                 75.35ms       ×17
     16384       6.25s      122.36ms       ×51                       —

**The middle column is the same work and the right-hand one is twice it.** A
causal call walks the triangle our global layer walks — rounded up to a `BK`
block the same way ours rounds `reach` down to a tile, so "half" is the shape
rather than the exact count; a masked call walks the rectangle, which is what
mlx-vlm's own layer does. The 16384-token mask is 34 GB
and was not built. In bfloat16 the reference is faster again — 95.84 ms causal at
16384 — and that column is not the comparison to make, since nothing here runs
bfloat16 attention.

**In arithmetic rather than in ratios: 17.97 TFLOP/s against our 0.352.** One
global layer at 16384 tokens is 2.2 TFLOP either way. A2 measured ours doing it
at 352 GFLOP/s and called that decades under the machine; this says what the
machine actually gives a kernel of the same shape written the other way, on the
same part, in the same dtype, in the same afternoon. **51× is not a tuning gap.**

**What it would be worth, and it is arithmetic.** The two attention rows are
58.2 s of a 133.47 s prefill at 16384. Seven global layers at the reference's
causal figure are 0.86 s; what the 35 windowed ones would cost in that structure
is not measured here, and even charging them their present 14.14 s the prefill
would be about 90 s. Charging them nothing it would be about 76. **Neither
reaches mlx-vlm's 34.31 s**, because the two matmul rows are 72.3 s and this
changes none of them.

### Whether the fast structure can keep the bits

**It cannot, and the milestone stops here.** Every kernel change in this repo has
preserved bit-identical output and the recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` has never moved. The structure
that carries the 51× is `simdgroup_matrix`, and a hardware 8×8×8 matrix
multiply-accumulate sums its `k` dimension in an order the instruction defines and
this side does not choose. Our score is `simd_sum` over lanes walking the channels
in a fixed stride; theirs is whatever the fragment does. **Those are different
floats in the last bits, on every score of every key.** `fast::exp2` against
`precise::exp` is a second one, on every weight.

**The terms that keep the bits are measured and they do not compound to
anything like it:**

- **occupancy**, 23 to 27% of the attention rows and 12 to 16% of the matmul
  rows, and bit-identical by construction — declaring different threadgroup
  memory changes no arithmetic at all;
- **the tile's two reductions**, 23 to 29%, and bit-identical if one thread
  reduces in the same order and broadcasts, which costs a barrier;
- **the band**, 9 to 28%, whose values are the same values however they arrive —
  materialised or derived — at the memory cost the paragraph above prices;
- **the dequantisation table**, 30% of both matmul rows, whose replacement would
  have to produce the same sixteen floats and could.

Taken together and generously those are about a factor of two on the attention
rows and a quarter on the matmul rows. **The 51× is on the other side of the
line**, and whether this project's core claim is worth crossing it for is a
decision that belongs to whoever owns the claim. **Nothing here crossed it, and
nothing here should be read as a recommendation to.**

### What this leaves for whoever caps the spans

**A prefill writes every key of a layer before that layer's one attention
dispatch reads any of them.** `hold(0, n)` reserves the whole span, the
projections fill all `n` slots, and then a single `encode_over` runs `n` query
rows over `n` keys. A windowed layer held in a ring of its window would therefore
have its early keys overwritten by its late ones *before* the dispatch that needs
them — so **capping a windowed layer means chunking its prefill**, which is a
second change beside the ring addressing and the rewind that "The keys a sequence
keeps" already names.

**And the ring cannot be 512.** `reach` rounds the window down to a 32-key tile
so that the bounded loop stays the unbounded one bit for bit, so a windowed row
reads up to `window + tile - 1` keys back — 536 of them for a row at position
599, which `a_query_row_walks_the_keys_its_window_and_its_position_leave_it`
pins. The cap is the window rounded up to a tile, and the ring's size has to be a
multiple of the tile for `from` and `to` to keep landing where they land now.

**The prize is unchanged and it is memory rather than time.** A windowed layer
already reads only its window: 698 GB/s and 24 µs a token at every length here.
Capping the span saves the bytes the table under "The keys a sequence keeps"
counts and buys no microsecond of this section's.

### What did not move, which is everything

Nothing here touches a dispatch: the block is given back and what a call
dispatches is the kernel `3e41885` left, byte for byte, with two measurements
beside it. The list below is a check on that claim rather than a comparison, and
it is what says so — its middle column taken while the block was in the tree at
a height of one, and its last after it came out:

    context     e7168ce     3e41885      here
       97         19.99       19.98     19.97
      385         21.34       22.21     21.36
      769         21.91       22.08     22.08
     2048         24.85       24.81     24.91
     4096         26.09       26.09     26.06
     8192         28.65       28.92     29.02

One sitting each and unpaired, which is the standard the two columns beside it
were taken to. The eight-token figures on the packed heads are **19.47 ms at
k = 0 against 19.44 and 15.96 at k = 2 against 15.96**, and `k = 4` is 1.011×
against 1.002 — still not comfortably worth running. The per-kernel prefill
table at 16384 tokens reads 44.06 s of global attention against A2's 43.74,
14.14 s of windowed against 13.99, 38.15 s of `packed_matmul_grouped` against
38.15 and 133.47 s of passes against 132.97 — a different sitting on a host that
drifts 1.7%, and the same table. The reads either side of it are the same reads:
30792198.52 MB issued by the 7 global layers against a 134 MB span apiece, at
699 GB/s. A block of four would have made that 7.7 TB; nothing shipped does.

**The prefill wall, one sitting a length, against the four lengths this file
records:** 12.83, 25.57, 54.32 and 134.42 s against 12.66, 25.36, 54.25 and
133.63 — 0.6 to 1.3% on a host with 1.7% of drift, taken while the block was in
the tree. Paired against `157fb6a` with the block back out, a 2048-token prefill
is 10670.5 ms of device time against 10670.7 over four pairs, ranges across and
two of the four falling either way, and a decode step 19.501 ms against 19.506
over seven with the device's own clock 18.673 against 18.682: **no claim on any
of the four rows, which is what the 0.4% coming off looks like from the other
side.** **The
reference was not re-measured**: nothing here changes what mlx-vlm does, so its
34.31 s at 16384 and the ×3.89 beside it stand as A2 took them, and a paired
cross-engine sitting was not spent to re-prove an arm that did not change.

**No token changed.** The recorded continuation is `[656, 13, 623, 180069,
86333, 60500, 220, 23]` and is what the device generates; the speculating cases
write the text one-at-a-time decoding writes at every depth they drive; and
acceptance is unmoved to the digit — **85% at k = 1, 87/78% at k = 2**, 85/65/55%
and 82/65/53/47% below that — the packed heads' own recorded row, digit for
digit. `the_bounded_loop_is_the_unbounded_one_bit_for_bit`,
`a_query_row_walks_the_keys_its_window_and_its_position_leave_it` and
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` pass
unrelaxed, and so do all 585 tests of the run against a real checkpoint.

### What the diagnosis did not move, which is everything shipped

**No source outside a `#[cfg(test)]` module changed, and that is checkable rather
than asserted.** Every hunk in `attention.rs` and `matmul.rs` falls inside `mod
tests`; `testing.rs` is `#[cfg(test)]` at its declaration; the two additions to
`device.rs` and `kernel.rs` are read-only accessors on `MTLDevice` and
`MTLComputePipelineState` that nothing in a forward pass calls. So the kernels a
prefill and a decode step dispatch are byte for byte the ones `58d3ae7` left, and
every mutant these tables price is a second pipeline compiled beside them and
thrown away.

**The arbiter is the in-model row and it is where it was.** One sampled
16384-token prefill this sitting, against the two the file already records:

    kernel                     here      A3      A2
    global attention         43.78s  44.06s  43.74s
    packed_matmul_grouped    38.15s  38.15s  38.15s
    packed_matmul_rows       34.24s  34.16s  34.16s
    windowed attention       13.99s  14.14s  13.99s
    every pass              133.02s 133.47s 132.97s

A 128.39 s wall in 1117 dispatches over 43 submissions, and the reads either side
are the same reads — 30792198.52 MB issued by the 7 global layers at 703 GB/s.

**The decode row, one sitting and unpaired, against what this file records:**
19.63, 21.19, 21.84, 24.74, 25.93 and 28.53 ms at 97 to 8192 against 19.97,
21.36, 22.08, 24.91, 26.06 and 29.02 — every figure inside the 1.7% this host
drifts, and all six on the fast side of it, which is what a sitting rather than a
change looks like. **585 gated tests pass and 42 are skipped where 38 were**: the
four are this milestone's own measurements, which need a clock and no checkpoint.

**The reference was not re-measured end to end.** `sdpa-probe` is one dispatch at
our own shapes and makes no cross-engine claim; mlx-vlm's 34.31 s at 16384 and
the ×3.89 beside it stand as A2 took them.

### Which of the questions this could not settle

**Four candidates were on the list and two of them are still open.**

**Threadgroup-memory bank conflicts are not measured.** The occupancy sweep rules
them out *of itself* — the ballast is declared last, so every address the walk
uses is where it was and only the total size moves — but that is an argument
about the instrument rather than about the kernel. What `scores` and `staged`
cost in conflicts on their own is unasked here.

**Serialisation on a dependency chain is measured only obliquely.** The barrier
arm is 4% and the two-reduction arm is 23 to 29%, and the second of those removes
a dependency and its instructions together; nothing here separates the two.

**Launch and drain is ruled out by arithmetic rather than by a new
measurement.** An empty dispatch of this kernel's grid is 1.4 to 1.9 µs by the
table under "What is left was diagnosed rather than attacked", against a
prefill-shaped dispatch of a second or more — so it is under a millionth of these
rows and no arm was spent on it.

**And the shape of the occupancy curve is a finding without a mechanism.** More
threadgroups a core is better down to a count and worse below it, on both
kernels, and whether the left arm of that U is a shared cache or a scheduler
limit is not settled here — only that the shipped kernels are both on the same
side of it.

### Taking the occupancy turn on the kernel that walks the keys

**The turn A4 priced and left is taken, and what it cost to get is the staging.**
A threadgroup of `fused_attention` is one query row of one head. Its four live
arrays are 3 KiB and the tile of values it stages before weighting them is 16, so
it declared 19 KiB; this part gives a core 80 KiB, so it held four threadgroups.
The sweep is now the declaration rather than a ballast beside it —
`how_many_threadgroups_of_a_prefills_attention_a_core_holds` compiles the walk
staging a tile and the same walk staging none, at every size either can be given:

    the values      a threadgroup   2048 global  2048 window   8192 global  8192 window
    where they lie       4.00 KiB      114.72ms      52.82ms         1.62s     236.66ms
    where they lie       7.00 KiB      114.80ms      52.86ms         1.61s     236.53ms
    where they lie       9.00 KiB      114.94ms      53.30ms         1.61s     239.89ms
    where they lie      11.00 KiB      114.65ms      52.72ms         1.61s     235.53ms
    where they lie      11.25 KiB      114.68ms      52.70ms         1.57s     234.68ms
    where they lie      11.50 KiB       71.65ms      34.66ms      924.40ms     152.83ms
    where they lie      12.50 KiB       71.60ms      34.60ms      923.95ms     152.84ms
    where they lie      13.00 KiB       71.64ms      34.66ms      923.92ms     152.85ms
    where they lie      13.25 KiB       71.50ms      34.63ms      923.36ms     152.80ms
    where they lie      13.50 KiB       77.14ms      36.95ms         1.03s     163.93ms
    where they lie      15.00 KiB       77.10ms      36.93ms         1.03s     163.88ms
    where they lie      19.00 KiB       92.51ms      43.88ms         1.26s     194.58ms
    staged              19.00 KiB       92.65ms      44.45ms         1.25s     196.71ms
    where they lie      23.00 KiB      120.88ms      57.79ms         1.65s     256.91ms
    staged              23.00 KiB      120.25ms      57.75ms         1.63s     255.99ms
    where they lie      31.00 KiB      177.69ms      84.76ms         2.45s     376.78ms
    staged              31.00 KiB      176.42ms      84.33ms         2.41s     373.81ms

**19 KiB against 19 KiB is the row the whole change rests on.** The two arms are
0.15% apart on the global column at 2048 tokens and 1.3% apart at the widest of
the four, on a walk whose values come from threadgroup memory in one and from
device memory in the other — so at prefill shape the staging buys nothing worth
16 KiB, and the whole of what it did to this row was declare them. The staged
rows stop at 19 because a threadgroup that stages copies a whole tile whatever it
declared: an arm below that is a walk writing past its own array rather than a
smaller staging, which is what makes those three rows three and not twelve.

**The turn is six threadgroups a core, and it is a plateau rather than an edge.**
A core's 80 KiB divides into six of any declaration in (11.43, 13.33] KiB and
seven of anything under, which is arithmetic; what the table measures is that the
row turns exactly there. 11.25 KiB is 114.68 ms where 11.50 is 71.65, and 13.25
is 71.50 where 13.50 is 77.14. The shipped figure is 12.5 KiB, about a kibibyte
inside each edge, so a compiler that rounds a declaration differently lands on
the same six. Against the 19 KiB the kernel declared, that is **22.6% at 2048
tokens and 26.1% at 8192** on a global layer.

**So the memory nothing reads is load-bearing, and it has to be memory nothing
reads.** A first attempt kept part of the staging — as many keys of a tile as the
turn left room for, which is real work rather than dead weight — and it does not
reach the turn: a walk that brings *any* of a tile in early reads 116 ms at this
declaration where one that brings none reads 71, and two staged keys cost as much
as nineteen. At the five-threadgroup declaration above it the two agree to 2%. So
what the left arm of A4's U punishes is the staging and not the memory, and why
is no more settled here than it was there. It is the mechanism A4 said to name
rather than tune around, met head on.

**Bit-safety comes before the number and is proven after it.** A value is the
same float whichever memory it was read out of, and each thread meets its tile's
values in the order it always met them; the tile itself cannot move, because
`TILED_VALUES` bounds it in both entries out of one Rust constant.
`a_value_weighted_where_it_lies_is_a_staged_one_bit_for_bit` is that on the bits
rather than on a tolerance — sixteen cases and 55.7 million elements, synthetic
and captured, driven unsplit and through an eight-way fold, `-0.0` and `0.0` two
different answers.

**And the decode path keeps the kernel it had, which is a predicate and not a
hope.** The turn was measured on both regimes before the line was drawn: with it
taken everywhere, a decode step's device clock falls 2.1% at 385 keys, 1.4% at
2048 and 1.3% at 4096 — and **rises 1.9% at 8192**, every pair the same way and
no ranges overlapping at any of the four. A prefill's gain is not worth a decode
step's long context, so the kernel is compiled twice out of one string and
`splits_for` picks: a call the span was not cut for reads its values where they
lie, a cut one stages its tile whole. The two answer the same bits, so what the
predicate decides is a rate and never an answer.

### The same turn on the two matmul rows

**A tile of the packed matmul is one simdgroup, shares nothing with the other
seven of its threadgroup, and declared no threadgroup memory at all** — so it ran
at whatever residency this part gives a kernel that asks for nothing, and A4
found that to be the wrong side of the same turn. There is nothing here a
declaration could hold: the working set is registers, no load is cooperative, and
the memory is dead by construction rather than by choice.
`how_many_threadgroups_of_a_prefills_packed_matmul_a_core_holds` now runs past
where the row stops improving:

    a threadgroup     q_proj, tiled   a routed bank, grouped
     0 KiB                   5.90ms                  19.53ms
     1 KiB                   6.01ms                  20.37ms
     4 KiB                   6.01ms                  20.37ms
     8 KiB                   5.52ms                  18.27ms
    12 KiB                   5.27ms                  17.13ms
    16 KiB                   5.23ms                  16.84ms
    20 KiB                   5.20ms                  16.63ms
    24 KiB                   5.18ms                  16.49ms
    26 KiB                   5.18ms                  16.50ms
    28 KiB                   6.00ms                  18.63ms
    32 KiB                   6.00ms                  18.65ms

**Monotone to three threadgroups a core and 13 to 16% worse at two**, which is
where A4's sweep stopped and could not have seen the far edge. Three of them is every
declaration in (20, 26.67] KiB; the shipped figure is 24. That is **12.2% and
15.6%** against declaring nothing, bought with memory nobody reads.

**Those two percentages are the boost clock's and the sustained clock's are 8.7
and 12.5** — see "Whether the occupancy turn survives a warm, order-reversed
re-run", which re-took this table warm and in both orders. The turn, its
declaration and its far edge are where this table puts them on either clock; what
a sweep this short cannot do is leave the boost window, and a prefill never
enters it. **The row above is kept as taken** because everything the rest of this
section says about `volatile` and about the decode path was established against
it.

**`volatile` is load-bearing here and was measured rather than assumed.** A
thread stores a zero to its own slot and loads it back on the next line, which is
exactly the shape a forwarding pass removes: without `volatile` the pipeline
reports *no threadgroup memory at all* whatever the source declares, and the
change is silently undone. `fused_attention` declares memory the same way and
needs no `volatile` for it — there the fill is a strided loop over one array and
the read is a different loop over another, with a runtime bound between them.
Neither kernel argues the point: each has a case that reads the bytes its own
pipeline reports, so a compiler that started or stopped folding either fails a
test rather than quietly giving the memory back.

**It is bit-safe because the zero is the zero.** A tile's running sums start from
the entry that was filled with `0.0f` where they started from the literal, and
nothing else about the walk moved — same operands, same order.
`a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers` is unchanged
and unrelaxed, and every arm of the sweep asserts the shipped answer as it goes.

**And it reaches no dispatch a decode step makes.** Only the two tiled entries
declare it, and `tiles` is false for every shape a decode step has — a single-row
projection, a two-row shared bank naming two experts, a six-row routed bank
naming six.

### What the two turns are worth, measured rather than projected

**A4's arithmetic was 24 s of a 133 s prefill and the two changes are 26.4 s of
it**, which is the first thing to say because the projection was labelled as one.
The arbiter is the in-model per-kernel row at 16384 tokens, one sampled prefill a
column:

    kernel                       A4   the keys   the matmul
    global attention         43.78s     31.83s       31.99s
    packed_matmul_grouped    38.15s     38.10s       33.19s
    packed_matmul_rows       34.24s     34.39s       26.50s
    windowed attention       13.99s     11.29s       11.44s
    every pass              133.02s    118.46s      106.63s

The two attention rows are 57.77 s and are 43.43; the two matmul rows are 72.39 s
and are 59.69. **Neither change moved the other's rows** — the matmul pair is
within 0.4% across the first column and the attention pair within 1.3% across the
second, on a host that drifts 1.7% — which is what says the two are independent
and that neither figure is carrying the other.

**Paired and alternating, with the order flipped each pair**, each change against
the commit before it:

    prefill        2048 before   2048 after   8192 before   8192 after
    the keys          12715ms      12239ms       54230ms      50813ms
    the matmul        12291ms      11245ms       50894ms      45915ms

Every pair moved the same way at every length and no two ranges overlapped; the
device's own clock moved with each — 10671 ms to 10277 and 10275 to 9126 at 2048,
49153 to 45697 and 45685 to 40756 at 8192.

**The wall a user waits, one sitting a length, against the four this file
records:**

    tokens        before        after     tok/s     mlx-vlm       gap
     2048        12.66 s      11.32 s     180.9      2.66 s     ×4.26
     4096        25.36 s      22.56 s     181.6      5.61 s     ×4.02
     8192        54.25 s      46.77 s     175.2     13.05 s     ×3.58
    16384       133.63 s     109.51 s     149.6     34.31 s     ×3.19

**The reference was not re-measured and its column is A2's**, for the reason A4
gave: nothing here changes what mlx-vlm does, and a paired cross-engine sitting
is not spent to re-prove an arm that did not move. The gap is ×4.76 to ×4.26 at
2048 and ×3.89 to ×3.19 at 16384 — **the first sitting in this file where the
long end is under three and a half**, and the curvature that arrives at the top
is smaller than it was: 8192 to 16384 is ×2.34 where it was ×2.46.

### What did not move, which is the whole decode path

**Every context, paired against the commit this milestone opened at**, on the
device's own clock, which is the arbiter for whether a change is the work:

    context   before    after                       claim
       97     18.469   18.407   ranges across, 1 of 3   no claim
      385     20.065   20.104   ranges across, 3 of 5   no claim
      769     20.772   20.689   ranges across, 2 of 3   no claim
     2048     23.445   23.477   ranges across, 2 of 3   no claim
     4096     24.868   24.832   ranges across, 3 of 3   no claim
     8192     27.443   27.474   ranges across, 1 of 2   no claim

That is what a predicate holding the line looks like from the outside, and it is
what the same table read *without* one that says the predicate was needed. **A
paired decode step could not be taken at a context before this milestone** — the
harness timed every one of them over the structured prompt's own 34 keys, which
is the one length nobody has — so `bench decode` takes a `--context` now and the
row above is what it is for. The unpaired table the timing tier prints reads
20.71, 20.99, 21.62, 24.56, 25.74 and 28.70 ms a token against A4's 19.63 to
28.53, one sitting each and inside this host's drift but for the 97-token row,
which the paired column above calls no claim.

**The eight-token figures, acceptance and the tokens a round did not move
either**, over five alternating pairs on the packed heads: 19.47 ms at `k = 0`
and 16.08 at `k = 2`, no claim at any depth on either clock, and 85% / 87-78% /
85-65-55% / 82-65-53-47% digit for digit with 1.829, 2.560, 2.909 and 3.368
tokens a round. **`k = 4` has stopped paying**, which the brief asked to be said
plainly: it reads 0.997× and 1.000× across the two sittings where A4 read 1.011×
and the milestone before it 1.002×, so it is a round that costs what it banks.

**No token changed.** The recorded continuation is `[656, 13, 623, 180069, 86333,
60500, 220, 23]`; all **590** gated cases pass and 42 are skipped, which includes
the cases asserting that 48 tokens of a longer prompt are byte for byte what they
are at `k` of 0, 1, 2 and 4 and that `--backend cpu` answers what it answered.
`the_bounded_loop_is_the_unbounded_one_bit_for_bit`,
`a_query_row_walks_the_keys_its_window_and_its_position_leave_it` and
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` pass
unrelaxed. The peak resident set is where it was and the span table under "The
keys a sequence keeps" is unchanged to the mebibyte: neither change allocates
anything, and threadgroup memory is not resident memory.

### What this milestone did not reach

**The occupancy turn was the largest bit-safe item A4 found and it is the only
one taken.** What is left is A4's list, with every share it quotes now a share of
a smaller row — the attention pair is 43.43 s where A4 measured these terms
against 57.77, and the matmul pair 59.69 s against 72.39. **So the shares below
are A4's, of A4's rows, and each is worth about a fifth more of the prefill that
is left than the number says.** Nothing was re-swept to say by exactly how much:
that would be a second run of A4's instrument rather than a change, and the
ranking it produced is what the list is for.

- **The dequantisation table**, 30% of both matmul rows then and about 36% of
  them now — 21.7 s of the 59.69 by that arithmetic, and the largest term nobody
  has attacked. Two gathers into a 16-entry constant array per packed byte, whose
  replacement would have to produce the same sixteen floats and could. *Attacked
  since, and the answer is no: "What each way of decoding a packed byte costs"
  below measures two replacements that produce exactly those floats, and both are
  slower than the gather.*
- **The tile's two serial reductions**, 23 to 29% of the attention rows. Every
  one of 256 threads walks a tile's 32 scores twice for two scalars. Bit-safety
  is available — one thread reducing in the same order and broadcasting is the
  same two floats — but it costs a barrier and trades 255 threads' issue slots
  for one thread's serial chain at the *same* latency, so whether it pays at all
  is a measurement rather than an inference and none was taken here.
- **The band it derives**, 9 to 28%, and **the keys and values**, 15 to 19%.
- **`exp` and `simd_sum` are still nothing** and still not worth a milestone.

**And the two questions A4 could not settle are still open.** Threadgroup-memory
bank conflicts are unmeasured. The U is still a finding without a mechanism — and
this milestone added a row to it rather than explaining it: at six threadgroups a
core a walk that stages any part of a tile is 63% slower than one that stages
none, at the same declaration, and nothing here says why.

### What each way of decoding a packed byte costs

**The largest untouched term is not a memory access, and the way to find that out
was to try to remove it.** A4 priced the decode by ablation — an integer-to-float
conversion where the two gathers were is 70 and 71% of the kernel — and read the
30% as the table's own latency, "a memory access nobody counts". That reading
licenses a replacement: MXFP4's sixteen values are one exact table, so any method
producing those bit patterns is the same decode, and one computed in registers
would owe nothing to memory at all.
`what_each_way_of_decoding_a_packed_byte_costs` compiles three that do, over the
same two shapes A4's limiter table is read across:

    the decode                            q_proj, tiled   a routed bank, grouped
    two gathers into a table — shipped           5.35ms                  16.52ms
    the field its bits assemble                  5.94ms                  18.77ms
    one gather into a table of pairs             5.75ms                  16.87ms

**The gather is the cheapest of the three and the arithmetic is the dearest**, by
11.0 and 13.6% — reproducible to the hundredth of a millisecond over two
sittings. E2M1 is f32's own layout an exponent field apart, so a code's value is
its three low bits laid where an f32 keeps its own and counted up from `0.5`:
about six integer operations, against one load of 64 bytes that every lane of a
simdgroup indexes separately. **A table of whole bytes is not the answer either.**
One gather into 256 pairs is half the loads the shipped decode issues and it is
7.5 and 2.1% slower — so what the 64-byte table costs is not the load, and 2 KiB
is already enough to lose whatever holds it.

**Measured end to end before it was withdrawn**, because a sweep over two
dispatches is not a prefill: the arithmetic decode was shipped and paired against
the commit before it over a 2048-token prompt, and it is **6.4% of the wall and
7.4% of the device clock** — 12076 ms against 11350, and 9798 ms of device time
against 9127, every pair the same way and the ranges apart. So A4's 30% is the
decode's dependency chain and the issue slots it takes, not the table it reads
through, and **the largest term on A4's list is not available this way.** What
ships out of it is the instrument and one inline function: `element` is the one
reading of the format the two entries now share, and it is free — 9126.6 ms of
device time against 9125.4 at the same prompt, ranges across, no claim.

**Bit-safety was proven before either number was taken and it is proven on the
bits.** The two arms that decode a *code* answer the sixteen floats through an
entry point of their own:
`every_way_of_decoding_a_code_answers_the_tables_sixteen_floats_bit_for_bit`
compares what the device wrote against `inkling_core::quant::ELEMENTS` as `u32`
patterns, which is what catches code 8's `-0.0` where `==` would call it code 0's
`+0.0`. The third decodes a *byte* and has no such entry point to give — a table
of pairs is indexed by both codes at once — so what stands in for it is the arm
every one of the three also passes:
`every_way_of_decoding_a_byte_multiplies_to_the_same_bits` puts each through a
whole multiply on both entries and compares the outputs the same way.

**And that is where a form this file would otherwise have shipped was caught.**
The arithmetic decode's first version needed no comparison at all: laying the
code's bits into an f32 and multiplying by `2^126` is exact for all sixteen,
because the code E2M1 calls subnormal lands on the f32 subnormal `2^-127`, which
is `0.5` under the same factor. **This device flushes f32 subnormals to zero.**
That form answered `0.0` for code 1 and `-0.0` for code 9 and was bit-exact for
the other fourteen — a decode wrong in one code of sixteen, on the two codes a
tolerance would never have found, caught by a probe that costs a dispatch.

### What reducing a tile in one thread costs, in two regimes that disagree

**A5 left the tile's two reductions as a measurement rather than an inference
and the measurement has two answers.** Every one of 256 threads walks a tile's
32 scores twice for two scalars each of them ends up holding — A4 priced that at
23 to 29% of the attention rows, the largest term of the walk after the keys.
One thread walking it and broadcasting is the same serial chain over the same
operands in the same order, so what the others read is the float they would have
computed; it costs no barrier inside the tile, because the maximum crosses on
the barrier already standing between the scores' last read and their first
overwrite, and the total is not wanted until the walk is over.
`what_reducing_a_tile_in_one_thread_costs` compiles both:

    the two reductions     a threadgroup   2048 global  2048 window   8192 global  8192 window
    every thread              12.50 KiB       71.60ms      34.64ms      923.26ms     152.73ms
    one thread, broadcast     12.52 KiB       67.58ms      34.10ms      856.63ms     148.49ms

**5.6 and 7.2% on the two global cells of that sitting**, and 5.4 to 6.1% on the
first of them across three — and 16 bytes of declared memory, which is the
column that says the arm did not move the occupancy instead of the reduction.

**In the model it loses, and the model is the arbiter.** The same arm inside a
769-token prefill, one sampled pass an arm:

    kernel                 every thread   one thread
    windowed attention         392.79ms     423.49ms
    global attention            88.75ms      95.99ms
    every pass                    3.41s        3.45s

That is **7.8 and 8.2% the wrong way**, and a paired 2048-token prefill agrees:
11273 ms against 11411 and 9126 ms of device time against 9395, every pair the
same way and the ranges apart. **So the arm was withdrawn and the kernel keeps
its 256 redundant walks.**

**What separates the two regimes is what else is running, and the bandwidth
column says so.** Alone, three rounds over one set of keys, the dispatch is warm
and issue-bound, and 255 threads' issue slots are worth having back. In the
model the same dispatch sits between two matmuls that stream a terabyte, and its
attention rows read **722 and 782 GB/s — 88 and 95% of this part's peak**. Under
a ceiling like that the redundant walks were already free, hidden behind memory
the kernel was waiting on anyway, and what the broadcast adds is a serialization
that is paid in full.

**This is a caution about the instrument and not only about the arm.** A4's
whole attention limiter table is taken on `a_prefill_costs`, a dispatch measured
alone, and this is the first arm carried across to the model and read there.
Nothing above is retracted — a term's share is still its share — but a share
measured warm is not a promise about a kernel running at 95% of peak, and the
three attention items left on A4's list are all arithmetic rather than traffic.
**The one thing that can move a row at that ceiling is reading fewer bytes.**

### What this milestone shipped, which is no kernel at all

**Two of the four items on A4's list were taken to a measurement and neither
paid**, so what the engine runs is what it ran: the packed matmul keeps its
gather and the attention kernel keeps its 256 redundant walks. The one change to
a source is `element`, an inline function the two matmul entries now decode
through instead of spelling the same gather twice, and it is free — 9126.6 ms of
device time against 9125.4 over a paired 2048-token prefill, ranges across.

**Nothing moved, and here is the check rather than the argument.** 593 gated
cases pass and 44 are skipped, which is A5's 590 and 42 plus the three cases and
two measurements this milestone adds. The recorded continuation is `[656, 13,
623, 180069, 86333, 60500, 220, 23]`; `--backend cpu` answers what it answered;
`the_bounded_loop_is_the_unbounded_one_bit_for_bit`,
`a_query_row_walks_the_keys_its_window_and_its_position_leave_it` and
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` pass
unrelaxed; the resident-set bound holds.

**The wall a user waits, one sitting a length**, against A5's column and the
reference's, which is A2's and was not re-measured:

    tokens        A5       here     tok/s     mlx-vlm       gap
     2048    11.32 s    11.31 s     181.1      2.66 s     ×4.25
     4096    22.56 s    22.14 s     185.0      5.61 s     ×3.95
     8192    46.77 s    46.08 s     177.8     13.05 s     ×3.53
    16384   109.51 s   108.90 s     150.4     34.31 s     ×3.17

Every length is at or just inside where A5 left it, which is what a milestone
that shipped no kernel should read.

**Decode, paired against the commit this milestone opened at**, on the device's
own clock: 27.275 ms against 27.316 at 8192 and 20.032 against 20.024 at 385,
ranges across, no claim at either. A first sitting at 385 read 20.085 against
19.986 with the ranges apart, which a second sitting did not reproduce — a
0.5% claim on a change that is one inline function is a sitting and not an
effect, and saying so is cheaper than believing it. The timing tier's unpaired
table reads 18.41, 19.63 and 20.37 ms at 97, 385 and 769 keys, all at or under
the figures A5 recorded.

**Speculation is where it was and `k = 4` has fallen further.** Acceptance is
84.8% / 87.0-78.3% / 85-65-55% / 82.4-64.7-52.9-47.1% and the tokens a round are
1.829, 2.560, 2.909 and 3.368 — digit for digit A5's. `k = 4` reads **0.972×**
in this sitting where A5 read 0.997 and 1.000 and A4 read 1.011, one unpaired
sitting against A5's five pairs. **Speculation's useful range keeps narrowing at
the top**, which nothing here targets and which is worth knowing: `k = 2` is
still the depth that pays, at 1.17×.

### What this milestone did not reach, and what the ceiling says about it

**The band it derives and the keys and values were not reached**, and the
reduction's two regimes say something about both. The attention rows inside a
prefill run at 88 to 95% of this part's peak bandwidth. The band is arithmetic —
a few operations per key against a small staged row — and the reduction is the
measured precedent for what arithmetic is worth under that ceiling: 23 to 29% by
warm ablation, and 8% the wrong way in the model. **The keys and values are the
traffic itself**, which is the one term a ceiling like that leaves open, and the
only way to move it is to read fewer bytes rather than to read the same bytes
better — which is a format decision and not a scheduling one.

So the list A5 left is now two items shorter by measurement and the two that
remain are ranked differently than the shares rank them. Neither was taken here.

**What was not measured, and why.** The per-kernel table at 16384 was not
re-taken: it is 110 s a reading before the sampling overhead, and no shipped
change touches a dispatch — the 769-token table was taken instead, on both arms
of the one experiment that needed an in-model reading. The paired decode sweep
was taken at two of the six contexts rather than all six, and the paired prefill
at 2048 rather than at all four lengths, for the same reason: what those would
be re-proving is a kernel nobody edited.

### What the matmul costs on the other side of the line

**A7 crossed the line A4 named and measured what is over there, and the answer
is that the matmul rows are 2.85 times faster.** Nothing below is a default —
see "Two numerics behind one flag" for what `--numerics` is and why the
reference is what a caller gets who does not ask. **Every figure here is the
production path against the reference path, in this engine, at these shapes.**

The two entries are `mma_matmul_rows` and `mma_matmul_grouped`, and they are the
two tiled ones with the reduction carried by `simdgroup_multiply_accumulate`
instead of by a lane-strided walk under a `simd_sum`. A threadgroup is 32 rows of
64 columns over eight simdgroups laid two down and four across; a step brings in
32 rows of the input and 64 columns of the weight, decoded, and drives them
through 8×8 fragments. **The weight is decoded once for the whole block where the
reference tile decodes it once for four rows** — so the decode's dependency
chain, which "What each way of decoding a packed byte costs" measured at 30% of
that kernel and could not remove, is amortised eight times as far. A step is
`GROUP_SIZE` codes wide, which puts exactly one scale byte under it and lets the
scale be folded in at the staging: a scale is a power of two, so a code times its
scale is exact and the multiply that follows owes nothing per step.

**Paired and alternating, in-model, the order flipped each pair:**

    prefill        reference   production   change   pairs
     2048, wall     11287ms       7127ms    -36.9%   5 of 5, ranges apart
     2048, device    9125ms       5051ms    -44.6%   5 of 5, ranges apart
     8192, wall     46424ms      28787ms    -38.0%   3 of 3, ranges apart
     8192, device   40784ms      23561ms    -42.2%   3 of 3, ranges apart

**Per kernel at 16384 tokens, one sampled prefill a column:**

    kernel                  reference   production
    global attention          33.09s       31.87s
    the grouped matmul        33.97s       11.81s
    the row-tiled matmul      27.15s        9.61s
    windowed attention        11.63s       11.28s
    every pass               109.49s       67.61s

**The two matmul rows are 61.12 s and are 21.43 s**, which is 2.85× and 65% off
them. At 8192 the same two rows read 16.06 and 13.01 against 6.13 and 4.81, which
is 2.66× — so the ratio holds within a tenth at half the length.

**The two attention rows did not move the way the matmul rows did**, which is
what says this change is the matmul's and is not carrying anything else: 33.09 s
of global attention against 31.87 and 11.63 of windowed against 11.28. That is
3.7 and 3.0% *down*, which is more than this host's 1.7% drift and is one sampled
prefill a column — the likely reading is that a 68 s pass carries less sampling
overhead than a 109 s one, and **no claim is made on those two rows either way**.
What matters about them is the sign: they moved a few percent where the rows
beside them moved by a factor of three.

**The wall a user waits, one sitting a length**, against the reference engine's
column, which is A2's and was not re-measured:

    tokens   reference   production    mlx-vlm   gap before   gap after
     2048     11.24 s      7.06 s      2.66 s      ×4.23        ×2.65
     4096     21.99 s     13.30 s      5.61 s      ×3.92        ×2.37
     8192     46.58 s     29.34 s     13.05 s      ×3.57        ×2.25
    16384    109.82 s     74.72 s     34.31 s      ×3.20        ×2.18

**This is the first arrangement in this file where the gap is inside two and a
half at every length.** It is also where the shape of the remaining gap changes:
at 16384 the two attention rows are now 64% of the passes where the two matmul
rows were 56%, so the half A4 under-sold for three milestones has stopped being
the larger one.

**The block is not free at every shape, and the sweep says where it turns.** A
block computes its 32 rows whether the call brought them or not, and a call of
one or two blocks does not put threadgroups enough on this part to fill it. On
the device's own clock, one sitting a length:

    rows   reference   production
      40     264.6ms     278.8ms
      48     315.0ms     320.7ms
      64     415.1ms     390.1ms
      80     518.2ms     487.3ms
      97     502.8ms     456.8ms
     385    1670.1ms    1357.4ms
     769    3355.8ms    2699.5ms

So a call is given a block only at two blocks' worth of rows, and **padding alone
does not explain the turn** — 48 rows waste a third of two blocks and 97 waste a
third of four, and only the second wins. What the short calls are also short of
is threadgroups: 48 rows of `q_proj` are 128 of them against 240 slots on this
part's 80 cores, where 97 rows are 256 and fill it.

**What the floor is worth is a speculative round, and it was measured before the
floor existed.** A verify block is the depth plus one rows through every
projection, so at a depth of three it clears `tiles`'s four-row bar and lands on a
block eight times too tall: `k = 3` read **37.33 ms a token against the
reference's 17.08** and `k = 4` 36.30 against 19.98, where `k` of 0, 1 and 2 were
untouched because their blocks are under four rows and never leave the untiled
entry. With the floor drawn, `k = 3` is 16.27 against 16.41 and `k = 4` 18.86
against 19.70. **A block that is faster on a prefill and twice as slow on a
speculative round is one finding and not two**, and it is the same finding
`splits_for` already carries: a shape predicate is what keeps a prefill's gain
off a decode step's throat.

### Whether the two paths' tokens ever part company

**They never did, over 384 sampled argmaxes, and the recorded continuation is
reproduced on the production path.** There is no array of bits to hold that path
to and there cannot be one, so what stands in for the oracle is the reference
path itself — two GPU implementations sharing every tiling decision, every
predicate and every dispatch, differing only in how the innermost sum is
carried. `just diverge` is that instrument.

    prompt                          tokens   generated   agreed
    enumeration, a chat turn            71          64       64
    prose, mid-sentence                 68          64       64
    code, mid-function                  77          64       64
    a chat turn with four asks          83          64       64
    a list of 573 primes              2123          64       64
    a factual question                  64          64       64

**Six texts rather than six lengths, and that is the corpus's whole design.**
Whether two accumulations name the same token is decided by how close the top two
logits are, and how close those are is a property of the text — the acceptance
study measured 99.7% at the first head on enumeration against 44.9% on prose — so
one prompt tiled to six lengths would report the agreement of whichever regime it
happened to be. **Length is the second axis and it decides whether a prompt
reaches the flag at all**: a call under two blocks' worth of rows runs the same
kernel under both words, so `bench diverge` refuses a corpus member under
`PackedMatmul::SHORTEST_BLOCKED_CALL` rather than reporting a thing agreeing with
itself. The list of primes is the one member long enough to reach the grouped
entry as well, which is about 1366 tokens.

**What is reported is leading agreement and not a count of differing tokens.**
Two free-running generations that part at step 12 have nothing comparable after
step 12, because each is continuing a different sentence by then.

**And the gated tier passes under the production numerics**: 619 cases, 44
skipped, `INKLINGRS_NUMERICS=production cargo nextest run`. **What that variable
reaches is worth stating exactly**, because the run passing is weaker than it
sounds: it is read by `real_checkpoint.rs`'s own device harness, so the cases
that stand a stack up on the GPU run the production entries, and the ones that
drive the binary as a subprocess do not — those spawn it without `--numerics` and
get the reference, the way any caller does. The recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is in the first group: it comes
back off the device under either word, at the same 19.55 ms a token and the same
0.24 GiB peak resident set; `the_bounded_loop_is_the_unbounded_one_bit_for_bit`,
`a_query_row_walks_the_keys_its_window_and_its_position_leave_it` and
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` pass
unrelaxed. Acceptance is **84.8% / 87.0-78.3% / 85-65-55% / 82.4-64.7-52.9-47.1%**
and the tokens a round 1.829, 2.560, 2.909 and 3.368 — **digit for digit the
same under both**, which is what a flag that reaches no decode dispatch should
read. `k = 4` reads 0.983× under the reference and 1.031× under the production
path in this sitting, one unpaired reading each, against A6's 0.972: **it has not
fallen further.**

**These are the packed heads' figures and the checkpoint is what has to be said
beside them.** A8 read 91.3-73.9% at `k = 2` and 84.2-73.7-63.2% at `k = 3` with
3.048 tokens a round, took the line above for stale, and the two runs are two
checkpoints: this one is `models/Inkling-Small-mxfp4-mtp4` and that one is the
default `models/Inkling-Small-mxfp4`, whose heads are the bfloat16 originals.
Both were re-run at `42effa1` on a host checked settled first, and **each
reproduces its own recorded row to the digit**:

    heads       k = 1     k = 2          k = 3            tokens a round
    packed      84.8%   86.96-78.26%   85-65-55%    1.829 2.560 2.909 3.368
    bfloat16    84.8%   91.30-73.91%   84.21-73.68-63.16%
                                                    1.829 2.560 3.048 3.368

`k = 4` is 82.35-64.71-52.94-47.06% on both. **Nothing moved.** `k = 1` and
`k = 4` are the two depths the format leaves alone and `k = 2` banks the same
2.560 either way — see "The gate is acceptance" below, which records both rows
side by side — so a run on the wrong checkpoint disagrees at exactly the two
depths that disagree and nowhere else. **Two figures parting and two agreeing is
the signature of a swapped checkpoint**, and it is the one shape that reads from
the outside like a line that drifted.

**With the flag at its default nothing moved at all.** A paired decode step is
19.765 against 19.946 ms at 385 keys and 27.420 against 27.384 at 8192, ranges
across and no claim at either — which is what the untiled entry never being
reached looks like from the outside.

### What the production path is not, which is more accurate

**It is the worse-conditioned of the two orders, and by a factor that grows with
the reduction.** A fragment accumulator carries the whole reduction as one
running sum — 4096 codes are 512 accumulate steps into the same register — where
the reference splits it 32 ways across lanes and reduces the partials in a tree.
Against an f64 accumulation of the same products:

    reduction   reference   production
           32     9.0e-8      1.6e-7
          128     9.5e-8      4.8e-7
          512     9.5e-8      7.4e-7
         2048     9.6e-8      1.4e-6
         4096     1.4e-7      4.1e-6

**That it is f32 noise at a reduction of 32 is what says the arithmetic is right
and only the chain is long** — four accumulate steps land within an ulp or two of
exact, and a kernel with the transpose or the staging wrong would be decades out
at every length rather than exact at the short one. Every product either path
forms is exact, because a code is one of sixteen table values and a group scale
is a power of two; nothing is rounded anywhere but in the adds.

**So "neither is more accurate" is the wrong summary of this pair and the right
summary of the flag.** The flag's claim is about checkability: one order is one
this side picked and `--backend cpu` reproduces, the other is the instruction's.
On this particular pair the reference also happens to be the better-conditioned
one, and 4.1e-6 of the peak output is still four decades under what would move a
token — which is what the 384 agreeing argmaxes say from the other end. **A
production path that summed in several accumulators would close most of that
gap** and is not built here.

### What the flag cost, which is eight files and about twenty lines of mechanism

**The maintenance surface is the question the design was drawn around**, so it is
reported rather than asserted. Eight files mention `Numerics` at all.
`numerics.rs` is 22 lines of mechanism and 20 of cases. `LayerKernels::compiling`
is the only constructor that takes the word and **only the matmul takes it from
there** — the norm, the convolution and the argmax have no reduction a matrix
instruction could carry, so a kernel that does not take the flag is a kernel both
paths run. Everything above the accumulate is shared: the tiling decisions, the
submission structure, the grouping's two ends, `splits_for`, both occupancy
turns, KV handling. The two production entries take the same six bindings from
the same encoder, read the same `Shape`, and are chosen by the same predicates at
the same shapes.

**Where the surface actually went is the command line and the cases**, not the
engine: of the 365 non-comment lines the flag commit added, `args.rs` and
`bench.rs` are 226 of them and most of those are the cases holding the refusals —
`--numerics` on `--backend cpu`, an unknown word, the order the two words arrive
in. The other 139 are spread over six engine files, and about half of those are
cases too: the mechanism itself is `numerics.rs`'s 22 lines, one extra
constructor apiece on `PackedMatmul` and `LayerKernels`, and a word threaded
through `backend::open`.

**One entry point is public that would not otherwise be**:
`PackedMatmul::SHORTEST_BLOCKED_CALL`. A differential corpus that cannot ask
where the flag's own line is would report perfect agreement between a thing and
itself, and would keep reporting it after the arithmetic behind the flag had
changed.

### What A3's query block reads now, which is a question this could not ask

**The retest was proposed as cheap and it is not, and that is the finding.** A3's
block (`c2095cb`, reverted in `58d3ae7`) was measured before A5 changed the
occupancy regime, and the brief for this milestone read "the code exists and
re-running it is cheap". Reverting the revert onto this tree conflicts in
**eleven regions of `attention.rs`**, and they are not incidental: A3's block
parameterises `source` by a block height where A5's parameterises it by a staging
and a residency, A3 compiles one kernel where A5 compiles two and picks between
them with `splits_for`, and the walk itself is rewritten in both. **Resolving that
is writing a kernel that has both, not re-running one that had one** — and a
resolution that is not bit-identical to what ships measures nothing at all, which
is the whole property A3's block was interesting for.

**What the two existing sweeps say when read together, which is arithmetic and
labelled as such.** A3's block declared about 3 KiB of live arrays a row beside a
16 KiB staging, so a height of four declared about 28 KiB. A5's sweep prices that
declaration on its own: 23 KiB is **120.88 ms** at 2048 global against 12.5 KiB's
**71.60**, which is a factor of 1.69 before any block exists. A3 measured height
four at ×0.72. **So the whole of A3's refusal sits inside what its declaration
alone costs at that height**, and A3's numbers cannot separate the block from the
residency it spent. That is a reading of two tables and not a run, and it says
the question is open rather than that the block would pay.

**What it would take to close it** is a block built on A5's kernel rather than on
A3's — the staging predicate, the two compiled regimes and the occupancy
declaration kept, the query rows blocked on top — which is a milestone and not a
retest.

### What this says about the instrument, which is the third thing it has said

A6 found that isolated and in-model readings can **disagree in sign**, and warned
that A4's whole attention limiter table rests on the isolated kind. This
milestone adds two readings to that account and neither is a contradiction.

**The first is that a shape predicate can invert a kernel's sign without the
kernel changing at all.** The same two entries are 2.85× faster on a prefill's
matmul rows and 2.2× *slower* on a speculative round's, and the only thing that
differs is how many rows the call brought. So "is this kernel faster" is not a
question a kernel has, and a sweep that answers it at one shape has answered it
at one shape. A6's caution was about the *regime* a dispatch runs in; this is
about the *shape* it is handed, and the two compound.

**The second is that the largest term is not the one the shares rank first, for
the second milestone running.** A4 priced the decode table at 30% of the matmul
rows and A6 measured two replacements for it and withdrew both, on the reading
that the 30% is the decode's dependency chain rather than the table it reads
through. **That reading is what this milestone confirms from the other side**:
the block does not replace the decode, it runs the same two gathers into the same
64-byte table — and it is 2.85× faster because it runs them once for 32 rows
rather than once for four. The term was never removable and was always
amortisable, and nothing in A4's ablation table could have said which.

**And a bandwidth column moved without a dispatch moving.** The block's arrival
made `PackedBank::moves`'s grouped bound thirteen times too loose and printed a
routed bank at 2378 GB/s — 290% of this part's peak, and so visibly a bound
rather than a rate. It was tightened by the block's own predicate rather than by
a measurement: a block is dispatched at a block's worth of runs an expert, so it
spans at most two of them and can name at most two weights, where a tile of four
rows really can hold four runs. **The rows the production matmul now declares put
it at 166 and 138 GB/s, 20 and 17% of this part's peak** — which is the column to
read next, because it says the matmul is not near the memory and the reference's
attention rows at 88 to 95% are.

### What this milestone did not reach

**Attention was not built, and Part 2 justifies building it.** The brief made it
conditional on the matmul result and the matmul result is 2.85× on its two rows,
so the condition is met and the work is not done here. (A8 did it — see "What the
attention step costs on the other side of the line", where the two rows this
paragraph leaves at 43.15 s read 2.25 s.) What it needs is not the
matmul's shape: `steel_attention.h`'s structure is a block of query rows through
fragments with the online softmax kept in registers and reduced by two
`simd_shuffle_xor` steps, and putting that on this kernel means rebuilding the
band derivation, `reach` and `last` per query row, `splits_for` and both compiled
regimes around it. **At 16384 the two attention rows are 43.15 s of 67.61 s of
passes**, so it is now the larger half — and A4's own probe says the reference
runtime does a global layer at 122.36 ms where ours takes 4.55 s.

**The accumulator chain was not split.** The production path's drift grows with
the reduction because one fragment accumulator carries all 512 steps; several
accumulators summed at the end would shorten each chain by that factor at the
cost of registers. It is a change to four lines of the kernel and it was not
measured.

**What was not measured, and why.** The per-kernel table was taken at all four
lengths on both paths and only two of them are quoted, because the four are one
ratio and the two ends say it. The paired prefill was taken at 2048 and 8192
rather than at all four — 16384 is 110 s an arm and seven pairs of it is forty
minutes to re-prove a ratio the other two lengths already agree on to a point,
and the four-length wall table above is one sitting a length instead. The paired decode was taken at two of the six
contexts, because the flag reaches no dispatch a decode step makes and the two
taken are the ends of the range. The cross-engine column was not re-measured, for
the reason A4 and A5 gave: nothing here changes what mlx-vlm does.

### Whether the production path should be the default, which is no

**The numbers are above and the reasoning is here, kept apart on purpose.**

It should not, and not because of anything in the measurements. Every number
this milestone took says the production path is better: 2.85× on the two matmul
rows, 37 to 45% off a prefill's wall and device clock at both lengths it was
paired at, nothing at all on the decode path, 384 of 384 argmaxes agreeing, the
recorded continuation reproduced on the device, the whole gated tier passing, the
same peak resident set, acceptance digit for digit. If the question were "is it
faster and does it change the answer", the answer is yes and no.

**The question it fails is a different one.** This engine's core claim — the one
every milestone in this file has been written under — is that a kernel's answer
is the CPU path's bit for bit, and that claim is what has made four milestones'
worth of mutations falsifiable. It is why `element` could be swapped for two
other decodes and *proven* the same floats; why the occupancy turn could be taken
on sixteen cases and 55.7 million elements rather than on a tolerance; why A3's
block could be refused on a timing rather than argued about. **Making the
production path the default retires that instrument for the engine's dominant
kernel**, and what replaces it is a differential run: 384 argmaxes over six
prompts, which is a good instrument and is not the same instrument. A tolerance
cannot catch what `-0.0` against `0.0` catches.

**Second, 384 tokens is not a bound on anything.** The two paths agreeing over
one corpus says the drift did not reach a coin-toss step in that corpus. The drift
is real and grows with the reduction — 4.1e-6 at the length every projection in
this checkpoint has — and every generated token is one argmax over it. A default
is a claim about every prompt anyone will ever run, and what is measured here is
six.

**Third, the flag's cost is already paid and the option is worth keeping open.**
Eight files, about ninety engine lines, and a differential harness that will be
worth more once there are two things behind the flag rather than one. Nothing is
gained by promoting it that is not already available to whoever types the word,
and what is lost by promoting it cannot be got back cheaply.

**What would change the recommendation**, in the order it would arrive: an
attention kernel behind the same flag, so that the choice is about the whole
engine rather than about one of its two halves; a differential run over a corpus
large enough to bound the disagreement rate rather than to fail to find one; and
the accumulator chain split, so that the production path's drift stops growing
with the reduction. **Two of those three are measurements and one is four lines
of kernel.** Until then the default is the reference, `--numerics production` is a
word anyone can type, and this section is the number the decision belongs to
whoever owns the claim.

### What the attention step costs on the other side of the line

**A8 built the kernel A7's conclusion asked for, and the answer is that the two
attention rows are nineteen times faster.** Nothing below is a default — see
"Two numerics behind one flag" for what `--numerics` is and why the reference is
what a caller gets who does not ask. **Every figure here is the production path
against the reference path, in this engine, at these shapes.**

The entry is `mma_attention`, and what it is is a threadgroup carrying a *block*
of query rows rather than one. The shipped entry gives a threadgroup one query
row of one head and scores a key with a lane-strided dot under a `simd_sum` —
one multiply-add a lane a channel, and a cross-lane reduction behind every
score. The block makes both of the step's multiplies matrix instructions: the
scores of 64 query rows against 32 keys, and the values weighted by them, so an
instruction carries 512 multiply-adds where the other carries one.

**What the block buys beyond the instruction is that a lane's query row does not
move.** The row is its simdgroup and its position inside an 8×8 fragment, and
neither term changes for the whole walk — so `tau`, the position and the
relative-feature row are read once and held, the online softmax is two registers
rather than threadgroup memory, and its two reductions are two
`simd_shuffle_xor` steps over the four lanes that share a row. The entry above
writes a tile's 32 scores to threadgroup memory and takes **four barriers a
tile** to reduce them; the block takes none.

**And the band is derived per score exactly as it always was.** What changed is
who derives it: the entry above computes it on lane 0 of a simdgroup while the
other 31 wait, which is why "the band it derives" is 28 to 34% of that kernel by
its own ablation. Here every lane derives the entries of its own two elements,
so the same `d_rel` multiplies a query-key pair are spread across the simdgroup
instead of serialised on a lane of it. `banded_entry` itself is one function
both entries call, which is what the flag's rule asks of everything that is not
the accumulate.

**Per kernel at 16384 tokens, one sampled prefill a column:**

    kernel                  reference   production   change
    global attention          32.06s        1.51s      21.2x
    windowed attention        11.44s      734.98ms     15.6x
    the grouped matmul        33.18s       11.81s       2.81x
    the row-tiled matmul      26.50s        9.63s       2.75x
    every pass               102.02s       24.68s       4.13x

**The last row is the command buffers' own clock and the four above it sum past
it**, by the 4.7% of over-reporting the sampling costs and the table's own footer
states — so the shares here are read against the summed passes and the ratios
against each other, which is what makes the columns comparable at all.

**The two attention rows are 43.50 s and are 2.25 s**, which is 19.4× and 94.8%
off them. **It generalised to the windowed layers and by a factor within 27% of
the global one's** — 15.6× against 21.2× — which is the answer to the
question A4 left open about whether a block is a global-layer lever. It is not:
35 of this checkpoint's 42 layers stop at a 512-key window, and a block of query
rows amortises a key read across its rows whether the span it walks is the
prompt or the window.

**The two matmul rows did not move the way A7 left them**, and that is worth
stating because it is the same two entries: 2.81× and 2.75× here against A7's
2.88× and 2.83× on the same dispatches at the same length. They are the flag's
other half and this milestone did not touch them.

**The wall a user waits, one sitting a length**, against the reference engine's
column, which is A2's and was not re-measured:

    tokens   reference   production    mlx-vlm   gap before   gap after
     2048     11.27 s      5.68 s      2.66 s      ×4.24        ×2.14
     4096     22.10 s      9.35 s      5.61 s      ×3.94        ×1.67
     8192     46.37 s     17.20 s     13.05 s      ×3.55        ×1.32
    16384    109.11 s     33.33 s     34.31 s      ×3.18        ×0.97

**At 16384 tokens this engine is faster than the runtime it has been measured
against since A2**, by 3%, and that is the first line of this file of which
anything like it is true. It should be read with two things attached: the
mlx-vlm column is A2's and nothing here re-measured it, and the reference column
is what a caller gets. What the row says without either caveat is that the gap
closes as the prompt grows, which is the shape a quadratic term coming off
produces and is the opposite of every earlier row here.

**Paired and alternating, in-model, the order flipped each pair:**

    prefill        reference   production   change   pairs
     2048, wall     11319ms       5680ms    -49.8%   5 of 5, ranges apart
     2048, device    9126ms       3385ms    -62.9%   5 of 5, ranges apart
     8192, wall     46347ms      17260ms    -62.8%   3 of 3, ranges apart
     8192, device   40808ms      12194ms    -70.1%   3 of 3, ranges apart

The reference arm of those pairs reads 11.32 and 46.35 s against the 11.24 and
46.58 this file records, which is 0.7% and 0.5% — so the arm that must not have
moved did not.

### What the block is refused, and the measurement that drew the line

**A decode step is one query row and a block computes sixty-four**, so the
predicate mattered more here than it did on the matmul and it was landed before
any timing was taken. A7's warning is the reason: the same two matmul entries
were 2.85× faster on a prefill and **2.2× slower on a speculative round**,
because a verify block of four rows landed on a block of 32. A block here is
sixty-four rows against a decode step's one.

**Two things stand there and only one of them is new.** `splits_for` already
cuts the span of any call whose grid is short of the machine, and the block is
refused a cut call outright — so a decode step at any context somebody has is
turned away before the floor is consulted. The floor is what catches the shapes
`splits_for` leaves whole: a five-row verify block at 8192 keys is not cut, and
without a floor it would land on a block.

**What a block of query rows is worth at each height**, on the device's own
clock, the block against the reference entry at four spans:

    rows   n over n   over 385   over 2048   over 8192
       1      0.45x      1.03x       1.00x       1.03x
       2      0.46x      0.99x       1.02x       1.04x
       4      0.47x      1.04x       1.01x       1.02x
       5      0.48x      1.00x       1.13x       1.06x
       8      0.52x      0.87x       1.16x       1.11x
      12      0.71x      0.90x       1.19x       1.17x
      16      0.87x      1.44x       2.15x       2.20x
      20      1.09x      1.66x       2.32x       2.20x
      32      1.27x      2.31x       3.06x       3.02x
      64      1.95x      3.58x       4.92x       5.54x
     128      3.79x      5.96x       8.61x       9.59x
     385     11.06x      8.91x      11.69x      17.80x
     769      8.92x     10.33x      12.19x      16.36x

**The line is drawn at 32 rows, which is half a block, and it is drawn where
every shape agrees rather than where the first one turns.** The column that
turns latest is the one whose span is no longer than its own rows — a prompt of
twenty tokens — and every span a real prompt gives the kernel turns at eight to
sixteen. Under 32 the call stays on the reference entry, which is a rate and
never an answer.

**The first column is what the floor is for**, and it says the same thing A7's
speculative round said at eight times the block height: at one query row the
block is 0.45×. The columns to its right say why one guard was not enough — at
five rows over 8192 keys the block reads 1.06×, which is a shape that neither
loses badly nor is worth having, and it is exactly the shape `splits_for` leaves
whole.

**And the rows above 385 are where the table stops being about the floor.** At
769 rows over 8192 keys the block is 16.4× on one dispatch, which is the figure
the in-model table above arrives at from the other end.

### Whether the two paths' tokens part company with attention behind the flag too

**They never did, over 384 sampled argmaxes, with both kernels behind the
flag.** The corpus is A7's — six texts rather than six lengths, for the reason
that section gives — and `just diverge` is the same instrument with one thing
added: the gate that refuses a prompt too short to reach the flag now takes the
**larger of the two floors** rather than the matmul's alone. A prompt that
reaches one entry and not the other would report a thing agreeing with itself on
the half it never reached.

    prompt                          tokens   generated   agreed
    enumeration, a chat turn            71          64       64
    prose, mid-sentence                 68          64       64
    code, mid-function                  77          64       64
    a chat turn with four asks          83          64       64
    a list of 573 primes              2123          64       64
    a factual question                  64          64       64

**And the gated tier passes under the production numerics**: 622 cases, 47
skipped, `INKLINGRS_NUMERICS=production cargo nextest run`. What that variable
reaches is what A7 stated exactly — the cases that stand a stack up on the GPU
run the production entries and the ones that drive the binary as a subprocess do
not. The recorded continuation `[656, 13, 623, 180069, 86333, 60500, 220, 23]`
is in the first group and comes back off the device under either word.

### What the block's order drifts by, which is what the reference entry's drifts by

**A7's finding does not repeat here, and that is the most load-bearing number in
this section.** The matmul's production order is the worse-conditioned of the
two by a factor that grows with the reduction — 4.1e-6 against 1.4e-7 at 4096 —
because one fragment accumulator carries all 512 steps. A softmax accumulates
over the whole key span, which at 16384 is longer than 512, so the same question
had to be asked of a chain the context decides the length of.

Against an f64 accumulation that forms the whole score row and shifts it by its
own largest in one pass — neither entry's arithmetic, deliberately, since both
of them stream:

    keys   the entry   the block   between them
     512      9.4e-7      9.1e-7        5.8e-7
    2048      3.0e-6      3.2e-6        1.2e-6
    8192      1.3e-5      1.3e-5        2.4e-6
   16384      3.2e-5      3.2e-5        4.3e-6

**The drift grows with the context and it grows identically for both.** At 512
keys the block is marginally the *better* of the two; at 2048 the entry is; at
8192 and 16384 they are the same figure. So what the first two columns measure
is not the block's order at all — it is the **streaming softmax**, which both
entries run and which rescales a running total once per tile of keys. That term
is common to the flag's two sides and grows about 34-fold over a 32-fold key
span.

**So the online algorithm's rescaling neither helps nor hurts the block relative
to the entry**, and the fragment accumulator that cost the matmul a factor of
thirty costs the attention step nothing measurable. The reason is in the shapes:
the matmul's chain is the whole 4096-long reduction in one accumulator, where a
score's chain is `head_dim` — 128 codes, sixteen accumulate steps — and the long
chain in attention is the softmax's, which is not a matrix instruction's to
carry. **A kernel behind this flag is worse-conditioned where the instruction
owns the long chain and neither better nor worse where it does not**, which is
one finding rather than two and is the sentence A7 could not have written.

### What did not move, which is the whole decode path

**Decode at every context, under both words, one sitting:**

    context   reference   production      peak, reference   peak, production
         97     20.05ms      19.99ms             0.25 GiB           0.25 GiB
        385     21.41ms      21.19ms             1.27 GiB           1.27 GiB
        769     21.91ms      21.66ms             1.28 GiB           1.28 GiB
       2048     24.77ms      24.82ms             0.97 GiB           0.97 GiB
       4096     26.29ms      26.08ms             2.74 GiB           2.74 GiB
       8192     28.73ms      28.56ms             4.32 GiB           4.31 GiB

Every cell is inside 1.2% and the peak resident set is the same figure at five of
the six contexts and a hundredth of a gibibyte apart at the sixth — which is what
a flag reaching no decode dispatch should read. The reference column is this file's own: 20.0 ms at
97 keys and 28.7 at 8192, to the tenth.

**Paired, the flag at each word:** a decode step at 385 keys is 20.080 against
20.093 ms of device time over seven pairs, ranges across and four of seven
falling the other way; at 8192 keys it is 27.606 against 27.629 over five pairs,
ranges across and three of five the other way. No claim at either, which is the
claim.

**The wall is the one row worth a caveat.** At 8192 keys three pairs read the
production side 3.9% slower with the ranges apart, and five pairs read 1.8% with
the ranges across and no claim. The device's own clock is flat in both sittings,
so whatever moved is not a dispatch; what it is is not attributed, and at five
pairs it is not an effect by this file's own standard.

**A speculative round is untouched at every block height**, which is what the
floor was drawn for:

    tokens a block    reference   production
        1              24.34ms      25.03ms
        3              37.69ms      37.22ms
        5              55.35ms      54.98ms
        9              80.81ms      80.19ms

A7's failure mode was `k = 3` at 37.33 ms against 17.08 before its floor existed.
Here the same depth reads 37.22 against 37.69 — the block never sees a verify
block, because nine rows is under both floors.

**Acceptance is digit for digit the same under both**: 84.8% at `k = 1`,
91.3-73.9% at `k = 2`, 84.2-73.7-63.2% at `k = 3` and 82.4-64.7-52.9-47.1% at
`k = 4`, with 1.829, 2.560, 3.048 and 3.368 tokens a round.

**Two of those figures disagree with the line this file records above, and the
reason is the checkpoint rather than the calendar.** This sitting ran the default
`models/Inkling-Small-mxfp4`, whose heads are the bfloat16 originals; the line
above was taken on `models/Inkling-Small-mxfp4-mtp4`, whose heads are packed. So
the nine figures here are the bfloat16 row and the nine above are the packed one,
and **both reproduce**: re-run at `42effa1` on a host checked settled first, each
checkpoint returns its own recorded row to the digit. A8 read this as the
recorded line having drifted and it had not.

**The two rows agree at `k = 1` and at `k = 4` and part only at `k = 2` and
`k = 3`, which is exactly what "The gate is acceptance" records the format
costing** — so two figures disagreeing and two agreeing is the signature of the
wrong checkpoint. It is worth naming because it is the failure this file's own
discipline does not catch: every number here was paired, alternating and
same-sitting, and none of that says which weights the sitting opened.

**`k = 4` reads 0.843× under the reference and 0.844 under the production path**,
against the 0.972 this file records. The commit before the block reads 0.858 in
the same sitting, so this is not the change — and **it is the same swapped
checkpoint and not the host**, which is what this paragraph originally said and
had wrong. Re-measured at `42effa1` on a settled host, the bfloat16 heads read
0.848× and the packed ones 0.960× at `k = 4`. The 0.848 is the figure these three
readings are of: it is 0.6% and 0.5% off the two taken across the block and 1.2%
off the 0.858 before it, where the 0.972 they were all held against is a seventh
away. That 0.972 is A6's and is a packed figure.

**The one loose end is that 0.848 is not the 0.832 "What the heads' format is
worth" records for the same checkpoint**, and 1.9% is above what this host
drifts. Neither figure is paired — that section's 0.832 is five alternating pairs
and this is one reading — so what the gap is of is not settled here. It does not
reach the argument: 0.848 against 0.843 and 0.844 is the comparison that decides
which checkpoint these readings came from, and it is tight where the comparison
against 0.972 is a seventh.

### What this sitting's absolute figures are worth, which is less than its ratios

**A decode step on this host today is 32.8 ms of wall against the 19.8 this file
records, and the commit before this milestone reads 32.775.** Both arms of every
pair above were taken in that state, so every *ratio* here is sound and every
absolute decode figure is this sitting's rather than this file's. The device's
own clock has moved less — 20.1 ms against 18.2 — which puts most of the drift
on this side of the round trip.

Nothing here diagnoses it. What is worth recording is the method: the figure was
checked against the *unchanged* commit through the same paired harness before it
was reported, which is the only thing that separates "the host drifted" from "the
change regressed", and it is a check this file has not always made.

### What the staging turned out to be worth, against the guidance

**Apple's guidance for this family is that threadgroup memory used as a
software-managed cache of a device buffer is slower than reading the buffer** —
register, threadgroup and buffer data share one cache hierarchy, so a copy moves
nothing nearer the multiply and what it adds is the copy, its barriers and a
declaration that decides how many threadgroups a core will hold. The block has
two such copies available to it and they are not the same copy, so both were
measured rather than assumed.

The values are the guidance's own case: the second multiply wants them
`[key, channel]`, which is how they already lie. The keys are not: the score
multiply's right operand wants them transposed, and read where they lie a lane's
two elements are `head_dim` floats apart while the copy that lands them can be
coalesced. On the device's own clock, four arms over one dispatch:

    a call, staging          neither    the keys   the values      both
    2048 tokens, global        1.00x       1.14x        1.13x     1.12x
    2048 tokens, window 512    1.00x       1.08x        1.13x     1.08x
    8192 tokens, global        1.00x       0.99x        1.04x     0.98x
    8192 tokens, window 512    1.00x       1.03x        1.13x     1.07x
    declared                    0 KiB      18 KiB       16 KiB    18 KiB

**Staging nothing is the worst of the four arms at three cells of four, and
staging the copy the guidance most clearly rules out is the best at three.** The
exception is the same cell in both readings: at 8192 tokens on a global layer,
staging the keys and staging both are 0.99× and 0.98× — the one place the
guidance's own direction shows. The shipped block stages its values and reads its
keys where they lie: 16 KiB declared, 1.04 to 1.13× over staging neither, better
than staging both at every cell, and within a point of the best arm at the one
cell it does not win.

**What that does and does not say.** It does not overturn the guidance: the
arms differ by at most 14 points where the block as a whole is 15 to 21× over the
entry it replaces, so this is a tuning term and not the structure. What it says
is that "read it where it lies" is a rule with an exception on the operand whose
*layout* has to change, and that the exception here is the one the rule most
clearly names. Why the value staging pays is unmeasured; the value block is read
once per channel fragment and there are sixteen of them, so a copy read sixteen
times is the shape a cache would help, and the same is true of the keys.

### What the flag's surface grew to, which is one more file and 143 lines

**Nine files mention `Numerics` where eight did**, and the ninth is
`attention.rs`. `numerics.rs` is unchanged — 22 lines of mechanism and 20 of
cases — and so is every refusal on the command line. `LayerKernels::compiling`
is still the only constructor that takes the word, and it now hands it to two
kernels rather than one.

**What this milestone added to `attention.rs`, by role:**

    the kernel source, in Metal            165 lines
    the mechanism, in Rust                 143 lines
    the cases and the sweeps               377 lines

The 143 are the constants the fragment layout rests on, `FusedAttention::
compiling` and its `blocked` predicate, the grid and the byte accounting the
predicate also decides, and one `Cell` that only a sweep sets. Everything above
the accumulate is shared and unchanged: `splits_for`, the two compiled regimes
and their occupancy declaration, the band derivation, the KV span, the
submission structure, the nine bindings and the encoder that fills them. The
block takes the same `Shape` from the same encoder at the same shapes the same
predicates chose, and the only thing about the dispatch that differs between the
two words is how many threadgroups cover the call.

**One entry point is public that would not otherwise be**:
`FusedAttention::SHORTEST_BLOCKED_CALL`, for the reason
`PackedMatmul::SHORTEST_BLOCKED_CALL` is — a differential corpus that cannot ask
where the flag's line is would report perfect agreement between a thing and
itself.

### What this milestone did not reach

**The block reuses no fragment, and that is the largest named lever left.** Every
`simdgroup_multiply_accumulate` here is fed by a fragment loaded for it and used
once: 16 channel fragments against 4 key fragments for the scores, and 4 against
16 for the values, so the load-to-multiply ratio is 1:1 on both. A block twice as
tall would hold two query-row fragments per lane and use each loaded key and
value fragment twice, taking that ratio to 2:1 for 32 more registers a thread —
and it would double the floor, which is the trade and is why it is not taken
blind. mlx-vlm's own steel GEMM runs 2.7:1.

**The block is float32 on both operands and the reference runtime ships no
float32 attention kernel at all** — its element types are 16-bit with a float32
accumulator, which halves both the staged footprint and the traffic. That is
available behind this flag and was not taken, because it is a different claim:
this flag's two sides sum the same exact products in different orders, and a
16-bit operand rounds the product itself. It would have to be reported in the
conditioning table above rather than beside it.

**The threadgroup is 256 threads because the block is 64 query rows**, and
mlx-vlm runs the same structure at 64 to 128. A narrower threadgroup is a
shorter block and a lower floor, which is the direction the first column of the
height sweep wants; it was not swept, because the width is a host-side constant
as well as a source one and the sweep would have had to reach both.

**That obstacle is the taller block's too and the two are one piece of work** —
see "What the taller block would cost at its floor", which prices the floor half
of the trade and sizes the rest. The block's height reaches the grid, `reach_for`
and the byte accounting, so neither axis can be swept until it is a property of
the compiled entry rather than a constant.

**The bandwidth column is divided by 819 GB/s and should be divided by 723.**
819 is this part's specification and 723 is what a streaming read achieves on
this machine, so every "of peak" figure in this file is about 12% low. Nothing
here changed it, because changing it moves every such figure in this file at once
and that is a commit of its own rather than a line of this one. The block's own
rows read 322 and 254 GB/s at 16384 tokens, which is 39% and 31% of 819 and 45%
and 35% of 723.

**The 723 was itself a written-down number and it is now measured at 725** — see
"What a streaming read actually achieves", which is the commit of its own this
paragraph asked for, and which also says why a float at a time reaches only 598.
The two figures this paragraph already converted stand; the denominator is
`725e9` from that sitting forward.

**One row of the per-kernel table moved on a kernel the flag does not reach.**
`rms_norm` is 427.93 ms of the reference's 16384-token prefill and 125.11 ms of
the production one, over the same 168 calls moving the same 73295.31 MB. Nothing
else under it moved — `short_conv` is within 16%, `dense_matmul`, `swiglu`,
`moe_combine` and both routers within 1% — so it is not the clock. It is 0.4% of the reference pass and 0.5% of
the production one, and it is unexplained here.

**What was not measured, and why.** The paired prefill was taken at 2048 and
8192 rather than at all four lengths, for A7's reason: 16384 is 109 s an arm on
the reference side and a pair of it is three minutes, so the four-length table
above is one sitting a length instead. The per-kernel table was taken at 8192 and
16384 on both paths and only 16384 is quoted. The cross-engine column was not
re-measured, for the reason A4, A5 and A7 gave: nothing here changes what mlx-vlm
does — and it is the one column of the wall table that now decides a sign, so it
is the first thing to re-measure before that sign is quoted anywhere.

### Whether the production path should be the default, which is now a closer question

**The numbers are above and the reasoning is here, kept apart on purpose.**

**What A7 said would change the recommendation has happened, and it is the first
of the three.** A7's answer was no, and it named three things in the order they
would arrive: an attention kernel behind the same flag so that the choice is
about the whole engine rather than one of its two halves; a corpus large enough
to bound the disagreement rate; and the accumulator chain split so that the
drift stops growing with the reduction. The first is built. The other two are
not.

**The recommendation is still no, and one of A7's two reasons has got weaker
while the other has not.**

The reason that got weaker is conditioning. A7 had to report that the production
path was the worse-conditioned order by a factor of thirty at the reduction
every projection in this checkpoint has, and could only say that 4.1e-6 was four
decades under what would move a token. The attention block adds nothing to that:
its drift is the reference entry's to a digit at every span, because the long
chain in an attention step is the softmax's and the softmax is not a matrix
instruction's to carry. So the flag's conditioning story is now "one of the two
kernels behind it is worse-conditioned and the other is not", which is a better
story than A7 could tell.

**The reason that did not get weaker is the oracle, and it got heavier.** This
engine's core claim is that a kernel's answer is the CPU path's bit for bit, and
that claim is what has made five milestones' worth of mutations falsifiable —
why `element` could be swapped for two other decodes and *proven* the same
floats, why the occupancy turn could be taken on sixteen cases rather than on a
tolerance, why `the_bounded_loop_is_the_unbounded_one_bit_for_bit` is a case
rather than an argument. **Promoting the production path retires that instrument
for both of the engine's dominant kernels rather than for one.** A7 gave up the
oracle on 70% of a decode step's device time; this would give it up on 96.7% of a
prefill's passes as well. What replaces it is still 384 argmaxes over six
prompts, and that number has not moved while what it is being asked to cover has
doubled.

**And the third thing has changed the stakes rather than the argument.** At
16384 tokens the production path is 33.33 s against a reference path's 109.11
and against mlx-vlm's 34.31 — so what is behind the flag is no longer a
tuning-grade win on one half of the engine, it is the difference between this
project being three times slower than the runtime it copies and being level with
it. **That is a reason to take the remaining measurements, not a reason to skip
them.** A default is a claim about every prompt anyone will ever run; six
prompts is what has been measured, and the cost of being wrong is now larger
rather than smaller because more of the engine is on the other side of it.

**What would change the recommendation**, in the order it would arrive: a
differential corpus large enough to bound the disagreement rate rather than to
fail to find one, which is the one item A7 named that neither milestone has
done; the matmul's accumulator chain split, which is still four lines of kernel
and would leave the flag with no worse-conditioned kernel behind it at all; and
a re-measured mlx-vlm column, because the wall table above now turns on a figure
taken three milestones ago. **Two of those three are measurements and one is
four lines.** Until then the default is the reference, `--numerics production` is
a word anyone can type, and this section is where the decision belongs to whoever
owns the claim.

### The debt this milestone cleared

**`what_a_prefills_attention_is_bound_by` had been failing since A5 and it is
fixed.** Its weighting arm anchored on a loop written once at one indentation;
A5's staging rewrite made that loop two — one against threadgroup memory and one
against the device, because a threadgroup pointer and a device pointer are
different types and nothing can name both — so the anchor matched neither string,
the arm asserted, and nextest's fail-fast cancelled the 36 cases queued behind
it. A7 ran the tier with `--no-fail-fast` to get around it; the tier runs
properly now.

The arm was worth having: the term is **16 to 22% of the walk** on both kinds of
layer at both lengths, which is a row this table had been missing rather than a
row it had wrong. What keeps the pair from drifting apart again is a count
check — a partial replacement is the one failure `instead_of` cannot see, since
it asserts the anchor is there and replaces every match, so a re-indent of one
writing alone would hold the term out of one entry, leave it in the other, and
print a share for a kernel that has neither shape.

**And A3's query block cannot be cheaply retested, which is recorded here so
that nobody spends the afternoon again.** A7 tried: reverting the revert
conflicts in eleven regions of `attention.rs`, structurally, because A3
parameterises the source by block height where A5 parameterises it by staging
and residency, and A5 compiles two kernels where A3 compiled one.

**What this milestone answers of A3's question is most of it.** A3 asked whether
carrying a block of query rows through one tile of keys pays, measured height
four at ×0.72, and was refused. The block above is that question asked on A5's
kernel rather than on A3's — the staging predicate, the two compiled regimes and
the occupancy declaration kept, the query rows blocked on top, which is what A7
said it would take. The answer is that a block of query rows pays enormously and
that **A3's height was the wrong axis**: four rows is under every floor this
milestone measured, and the sweep above reads 0.47× at four rows on the shape A3
took. A3's refusal was correct at A3's height and says nothing about 64.

What A3's question this does *not* answer is whether a block pays on the
reference numerics, where the scores stay a `simd_sum`. Nothing here separates
the block from the instruction — they arrived together and the flag is what let
them.

## Re-measuring what nobody had re-measured

**This milestone shipped no kernel and that is what it was for.** Three of this
file's headline numbers rested on measurement that was stale, suspect, or taken
in a state nobody had checked: a cross-engine column three milestones old, an
occupancy turn whose effect is smaller than a known measurement artifact, and an
acceptance figure a milestone had reported as no longer reproducing.

**The acceptance one is settled where it was recorded rather than here** — see
"The gate is acceptance" and the two paragraphs correcting A7's and A8's readings
of it, which is that both rows reproduce and the two runs were two checkpoints.
The other two are below. **The one that moved is none of the three**: it is the
reference's decode step, which nobody had listed because nobody had doubted it.

### That the host was settled, which is the first thing and was not free before

**A8 reported a decode step at 32.8 ms against the 19.8 this file records, on an
unchanged commit**, and correctly declined to publish an absolute figure on it.
So nothing here was taken until a known quantity reproduced. `bench decode` at
`42effa1`, five consecutive runs:

    decode  19.441  19.407  19.435  19.433  19.486 ms
    device  18.618  18.599  18.628  18.579  18.650 ms

That is a 0.4% spread on both rows and it is the file's own figure rather than
A8's. **And the paired harness was then run against itself** —
`just bench HEAD~1 . decode`, seven pairs, where `HEAD~1` differs from the
working tree by test code alone:

                unit        a        b   change  ranges   pairs      claim
    decode        ms   19.447   19.440    -0.0%  across   2 of 7  no claim
    device        ms   18.630   18.610    -0.1%  across   3 of 7  no claim

**A null pair reading no claim is what says the instrument is not inventing
effects**, and it is the control this file has never printed. Whatever A8's host
was in, it is not this one; nothing here diagnoses it either.

### One clean cross-engine sitting, which is the column three milestones deferred

**A2 took the cross-engine table and A4, A5, A7 and A8 all declined to re-take
it**, each for the same defensible reason — nothing they changed reaches
mlx-vlm — and the effect was that a column deciding the sign of a comparison
went three milestones without being read. **Everything in this section says they
were right; the row that was wrong by a factor of three is three subsections
down**, under "The reference's decode step", and nobody could have known because
nobody looked.

**Seven pairs, one sitting, the order flipped each pair**, on the packed heads,
this engine at its default numerics: `just checkpoint=models/Inkling-Small-mxfp4-mtp4
bench-engines`. **All twenty-four readings are claims** — every pair the same way
with the ranges apart — where A2's sitting had two that were not.

    prompt × generated    ours k = 0   ours k = 2    mlx-vlm    k = 2 against it
     97 × 128                3.030 s      2.549 s    3.210 s          1.26× ahead
    385 × 128                5.880 s      5.647 s    3.675 s          0.65×
    769 × 128                8.419 s      8.225 s    4.189 s          0.51×
     97 × 512               10.923 s      8.474 s   12.193 s          1.44× ahead

**The reference did not move and that is now measured rather than assumed**:
3.210 against A2's 3.209, 3.675 against 3.673, 4.189 against 4.186 and 12.193
against 12.131 — four wall times three milestones apart, agreeing to 0.5%. So
every milestone that declined to re-measure this arm was right about the arm, and
the cost of being right about it is in the decode section below.

**The crossover moved in and the sign of every row is A2's.** At 97 tokens
speculating two deep we start 262 ms behind at the first token and take 7.27 ms
less per token after it, so the wall times cross at **about 36 generated
tokens** where A2 read 49; the `97 × 512` row puts it at 35 the same way. At 385
and 769 there is still no crossover at any depth, for A2's reason — our decode
step at those contexts is slower than the reference's, so every token widens the
gap.

**The prefill and the decode step out of the same seven pairs:**

    tokens        ours    mlx-vlm     gap      context   ours k=0  ours k=2  mlx-vlm
       97       541 ms     283 ms   ×1.91          97      19.60     15.78    23.05
      385      2481 ms     706 ms   ×3.51         385      26.76     24.33    23.38
      769      4858 ms    1170 ms   ×4.15         769      28.05     27.04    23.78
                                              97 → 609      20.32     15.50    23.31

A2 read ×1.98, ×3.87 and ×4.61 on the left; the reference's three figures are
within 0.4% of A2's and ours are what A5 and A8 left.

### What a prefill costs against the reference, both numerics, one sitting

**The long prefill is where this engine's last two milestones went and it had
never been weighed against a reference column taken in the same sitting.** Ours
one sitting a length, warm, `bench prefill`; the reference's
`just prefill-bench` in the same sitting. Not paired — the effects are decades
and this host drifts 1.7%, which is the standard the four-length table has always
been taken to:

    tokens    ours reference   ours production    mlx-vlm    production against it
     2048          11.206 s          5.563 s      2.66 s                  ×2.09
     4096          21.962 s          9.374 s      5.62 s                  ×1.67
     8192          45.685 s         17.145 s     13.05 s                  ×1.31
    16384         108.989 s         33.120 s     34.37 s          **×0.96 ahead**

The device's own clock under each: 9.126, 18.935, 40.753 and 100.371 s on the
reference path, and 3.385, 6.257, 12.174 and 24.600 on the production one.

**At 16384 tokens the production path is ahead of mlx-vlm, and this is the first
sitting in this file entitled to say so.** A8 reported 33.33 s against a
reference column of 34.31 that was A2's and flagged it as the first thing to
re-take; re-taken, the reference reads 34.37 s and the claim survives with the
sign it was reported at. **It is one sitting a length and not a paired one**, and
4% is inside what a paired sitting exists to settle — so what this says is that
the two are level at 16384 and that the direction is ours, not that a 4% win is
established.

**Ours reproduces both of the last two milestones' rows**: 11.21, 21.96, 45.68
and 108.99 s against A5's 11.32, 22.56, 46.77 and 109.51 on the reference path,
and 33.12 against A8's 33.33 at 16384 on the production one. **The reference
reproduces A2's to the hundredth** — 2.66, 5.62, 13.05 and 34.37 against 2.66,
5.61, 13.05 and 34.31.

**The gap closes with length on the production path and opens with it on the
reference one**, which is the whole of what the flag is now worth: ×2.09 to ×0.96
across an eightfold prompt, against ×4.21 to ×3.17 on the default path in the
same rows.

### The reference's decode step, which is the figure that did not survive

**"The reference takes a threefold step at 2048 and plateaus near 78 ms" does not
reproduce, and it is the largest correction this milestone makes.** Same script,
same default context list, same pinned mlx-vlm — `reference/uv.lock` has not
changed since the sitting that recorded it — and the peak-memory column comes
back **identical to the digit at all eight contexts**, which is what says the two
runs did the same work:

    context     recorded    now     peak recorded    peak now
       97          23.58   23.38        130.99 GiB   130.99 GiB
      385          23.67   23.88        131.94 GiB   131.94 GiB
      769          24.52   24.86        132.97 GiB   132.97 GiB
     2048          77.85   26.33        135.86 GiB   135.86 GiB
     4096          74.93   26.68        136.70 GiB   136.64 GiB
     8192          78.70   29.03        138.60 GiB   138.60 GiB
    16384          79.36   34.02        142.44 GiB   142.48 GiB
    32768          91.27   43.01        150.18 GiB   150.18 GiB

**The first three rows reproduce and the last five are two to three times
lower.** Identical allocations at every row and identical prefills, so the
workload was the same and the clock was not. Nothing here diagnoses what the
earlier host was doing; what can be said is that it is not reproducible on this
one, with 293 GiB free and swap at zero, over two runs an hour apart.

**So the reference is a gentle slope and not a discontinuity**: 23.38 ms at 97
keys to 43.01 at 32768, which is 0.60 µs a token of context over a 340-fold
range. It was never flat and it never took a step.

### Where the two engines' decode steps actually sit

**Both on their own recorded instruments, one sitting** — ours by
`what_a_decode_step_costs_as_the_context_grows`, the reference by
`context_sweep.py`, unspeculated:

    context      ours   ours device   mlx-vlm   ours ahead
       97       20.05         18.88     23.38        1.17×
      385       21.35         20.15     23.88        1.12×
      769       22.10         20.90     24.86        1.12×
     2048       24.91         23.67     26.33        1.06×
     4096       26.31         25.09     26.68        1.01×
     8192       28.69         27.42     29.03        1.01×
    16384           —         33.92     34.02        1.00×
    32768           —         43.74     43.01        0.98×

**"This engine is 2.7× ahead at 8192" is withdrawn and the honest figure is
1.01×.** Ours reproduces its recorded row to a tenth — 20.05, 21.35, 22.10,
24.91, 26.31 and 28.69 against 19.99, 21.34, 21.91, 24.85, 26.09 and 28.65 — so
what moved is the column beside it and nothing about this engine.

**The lead is real at a short context and it is gone by 4096.** 1.17× at 97 keys
narrows monotonically, the two are indistinguishable from 4096 to 16384, and at
32768 the reference is 1.7% ahead. **Both engines walk the span and both slopes
are shallow**; what this file had was one slope measured against a plateau that
was not there.

The last two rows are ours on the device's own clock against the reference's
wall, which flatters us by the 1.3 ms of host term the 8192 row shows between our
two columns — **so ours is if anything read generously out there and is still not
ahead**. Extending the recorded instrument past 8192 is what would make those two
rows a claim rather than a reading.

### What the peak resident set measures, which is not the same thing on both sides

**This file has implied a 32× memory win and the number is about right and the
sentence is wrong.** Sampled from outside by the same `ps -o rss` at an
8192-token context, in the same sitting: **ours peaks at 3.25 GiB and mlx-vlm's
at 131.54 GiB**, and MLX's own `get_peak_memory` reports 138.60 GiB for that run
— so the allocator figure this file quotes for the reference is its process's
too, to 5%, and the comparison is not an accounting artifact in the direction it
was feared to be.

**What it is not is 40× less memory to run in.** Ours is a resident set over a
checkpoint that is **mmap'd**: the weights are clean, file-backed pages the
kernel reclaims at will, and a decode step reads 5.9 GB of them. So 3.25 GiB is
what happened to be resident when `ps` looked, not the working set — the engine
touches far more than that every step and gives it straight back. The
reference's 131.54 GiB is anonymous allocation it holds for the process's life.

**The claim worth making is about where the weights live rather than how many
there are.** Both engines read the same 140 GB checkpoint; this one leaves it on
disk and lets the page cache decide what is resident, and mlx-vlm materialises it
in its allocator. On a machine with less memory than the checkpoint, that is the
difference between running slowly and not running — which is a real property and
is not a 40× smaller footprint. **The two columns of the sweep above are not the
same quantity and should not be divided.**

### Whether the occupancy turn survives a warm, order-reversed re-run

**It does, and the way it survives is worth more than the fact.** External work
reproducing the threadgroup-memory experiment could not reproduce "declaring dead
threadgroup memory helps": its own first sweep showed a 1.7× win that vanished
once it added thirty warm-up dispatches and swept up-then-down. That artifact is
*larger* than the 12 to 16% this file's matmul sweep reports, and its sign is
exactly the one that manufactures the result — so the sweeps now warm the device
for two seconds and run their arms up the list and then down it.

**No ordering artifact exists in any of the three sweeps.** Up and down agree to
a tenth of a percent at every arm of every one, which is the whole question and
the answer is flat.

The attention sweep reproduces A5's table digit for digit in both passes — 114.67
and 114.60 ms at 11.25 KiB against 71.59 and 71.64 at 11.50, the same sharp step
at the same declaration, and the shipped 12.5 KiB still **22.5% and 26.6%** ahead
of the 19 KiB it replaced at 2048 and 8192 tokens on a global layer, against the
22.6 and 26.1 A5 recorded. The staging table reproduces too, both ways: staging
the values is the best of the four arms at all four cells, and the exception A8
named is where A8 named it.

**The matmul sweep is the one that moved, and it moved in magnitude and not in
shape:**

    a threadgroup    q_proj, tiled    a routed bank     q_proj, warm    a bank, warm
     0 KiB                  5.90ms          19.55ms           5.98ms         19.68ms
     8 KiB                  5.52ms          18.32ms           5.53ms         17.64ms
    12 KiB                  5.27ms          17.14ms           5.48ms         17.50ms
    16 KiB                  5.23ms          16.84ms           5.45ms         17.36ms
    20 KiB                  5.20ms          16.63ms           5.44ms         17.23ms
    24 KiB                  5.18ms          16.49ms           5.46ms         17.21ms
    26 KiB                  5.18ms          16.50ms           5.46ms         17.21ms
    28 KiB                  6.00ms          18.63ms           6.28ms         19.41ms
    32 KiB                  6.00ms          18.64ms           6.27ms         19.43ms

**The turn is at the same declaration, the far edge is at the same declaration,
and what the shipped 24 KiB is worth against declaring nothing is 12.2% and 15.6%
cold against 8.7% and 12.5% warm.** The two left columns are the sweep with the
warm-up taken out, run as a control, and they reproduce A5's recorded row to the
hundredth — so the difference is the warm-up and nothing else about the sitting.

**Which clock a sweep reports is the finding, and it is not the artifact the
external work found.** A ramp would move the arms *by their position in the
sweep*, and both passes agreeing to a tenth of a percent rules that out on either
clock. What the warm-up moves is which clock every arm is on at once: an arm of
this sweep is about 300 ms and there are eleven, so the whole of it fits inside
this part's boost window, where a single row of the attention sweep is seconds
and is on the sustained clock by its second arm whatever it opened on. **That is
why one of the two moved and the other did not**, and it is why the attention
figures needed no correction.

**The warm column is the one to carry**, because a prefill of any length runs on
the sustained clock and never on the boost one. What it corrects is a diagnostic
rather than a claim: **A5's shipped changes were measured by paired, alternating,
in-model prefills**, which a clock state cannot reach — 12715 to 12239 ms and
12291 to 11245 at 2048 tokens, every pair the same way — and those stand. The
26.4 s of a 133 s prefill is a measurement and not this sweep's arithmetic.

**So the turn holds, with the ordering stated, and the number beside it is
smaller than the one this file carried on one of the two kernels.**

### What the taller block would cost at its floor, and why the other half is not measured

**The trade is a floor against a reuse ratio and only one of the two can be
measured without a refactor.** A8 left fragment reuse at 1:1 on both multiplies
where mlx-vlm's steel GEMM runs 2.7:1, and named the obstacle: a block twice as
tall holds two query-row fragments a lane and uses each key and value fragment
twice, **and doubles the floor** from 32 query rows to 64.

**The floor is the half that is measurable, and it is priced by
`what_a_block_of_query_rows_is_worth_at_each_height`** — what a call of 32 to 63
rows would lose by falling back to the reference entry:

    rows      own rows, global   own rows, window   2048 keys, global   2048 keys, window
      24                 1.08×              1.11×               3.14×               2.07×
      32                 1.26×              1.33×               3.02×               2.44×
      48                 1.01×              0.98×               3.83×               3.00×
      64                 2.06×              2.06×               4.98×               3.95×

**What the band costs depends entirely on the span the rows come with, and the
two columns disagree by a factor of three.** A call of 40 rows whose span is its
own 40 keys — a fresh short prompt — gets nothing from the block at all: 0.98 to
1.33×, which is inside what an unpaired reading settles. A call of 40 rows
against a 2048-key span — **a follow-up turn in a session that already has a
context** — gets 2.4 to 3.8×, and that is what doubling the floor would give
away. So the floor's cost is not "short prompts", it is short turns on long
sessions, which is the shape a chat server has and the shape none of this file's
benchmarks measure.

**And the premise that doubling the floor reaches a speculative round is
wrong.** A round of depth `k` verifies `k + 1` rows, which is 5 at this repo's
own `SWEPT` and 9 at the deepest the block table prices — every one of them
already under the floor of 32, and already refused twice over by `splits_for`
besides. A8's own reading records it: "the block never sees a verify block,
because nine rows is under both floors." **Doubling the floor moves calls of 32
to 63 rows and nothing else**, so speculation is not on either side of this
trade.

**The 48-row cell is anomalous in both arms and is not built on here**: 182 µs
against 98 at 64 rows on a shape that dispatches the same single block, with the
reference entry moving the same way. One unpaired reading, unexplained, and
flagged rather than smoothed.

**The reuse half was not measured, and the reason is the one A8 gave for the
thread width — they turn out to be the same obstacle.** `MMA_ROWS_A_BLOCK` is a
host-side constant as well as a source one: the grid is
`heads * queries.div_ceil(MMA_ROWS_A_BLOCK) * THREADS_PER_GROUP`, `reach_for`
cuts the call into blocks of it, and the byte accounting charges per block of it.
So an arm compiled through `FusedAttention::from_source` with a taller block
would be dispatched over a grid sized for the shorter one — **a wrong answer
rather than a slow one**, which is why no sweep here can reach it and why this
is sized rather than guessed.

**What it would take, named so the next milestone does not rediscover it**: the
height becomes a property of the compiled entry rather than a constant, threaded
through the grid, `reach_for` and `PackedBank`-style byte accounting; then the
kernel body carries `held`, `weighted`, `scores`, `peak`, `total`, the position
and the store over a lane's rows rather than one, which is 64 more registers a
thread. **That is a milestone and not a line**, and the floor table above is what
it should be held against: it has to buy more than 2.4 to 3.8× on a short turn
over a long session, which is the only workload it takes anything from.

### What a streaming read actually achieves, which is the denominator

**A8 left the bandwidth denominator as a written-down number replacing a
written-down number**, and the whole of this milestone's argument is that those
do not survive being asked. `what_a_streaming_read_achieves_on_this_machine` is
the friendliest shape this repo can arrange for the memory system — 4 GiB, read
once, in order, nothing kept:

    traffic                        moved      achieved
    one buffer read in order     4.0 GiB      598 GB/s
    one read and one written     8.0 GiB      650 GB/s
    the same read four wide      4.0 GiB      725 GB/s
    the same copy four wide      8.0 GiB      682 GB/s

**A8's 723 is right and is measured at 725**, which is 0.3% away and is the only
figure this milestone re-took that came back where it was put. **The width of a
lane's load is what decides it**: the same read a float at a time is 598, so 127
GB/s of this part's ceiling is reachable only by a kernel that asks for four
floats at once — a fact about the kernels the column describes and not only about
the column.

**This is the one sweep in the crate the warm-up does not move**, and that is
worth saying because it is the same discipline the occupancy sweeps needed: cold,
the four arms read 590, 651, 726 and 682 against these. A kernel bound by memory
is not bound by the core clock, so the clock state the matmul sweep turned out to
be about cannot reach this one.

`MEMORY_BANDWIDTH` is `725e9`, measured on whatever host the case runs on rather
than asserted about this one. **The "of peak" figures recorded above divide by
819 and convert by multiplying by 1.130** — every one of them except the two the
paragraph under "What this milestone did not reach" already converted to 723, and
those are within half a percent as they stand. The tables are not rewritten,
because rewriting several hundred percentages by arithmetic is how a file
acquires a figure nobody took.

### What did not move, which is everything the engine does

**No kernel changed and no default changed.** What this milestone touched is
three measurement cases, one test-only denominator and this file. The flag stays
defaulted to reference and A8's reasoning for that is untouched: promoting the
production path retires bit-for-bit checkability for 96.7% of a prefill's passes
against the same six prompts, and neither of the two things A7 named as
prerequisites has been done.

**All 672 cases pass against a real checkpoint** — 624 in the gated tier and the
48 of the timing tier, which **runs to completion without `--no-fail-fast`**, so
A8's re-anchored ablation arm is still anchored. **The gated tier passes under
`INKLINGRS_NUMERICS=production` too**, at the same 624 and 48 and in the same
532 s, which is what a milestone that changed no kernel should read as; one
argument-parsing case reported a leaked handle under that word and passed, on a
test that opens no device and reads no checkpoint. The recorded continuation
`[656, 13, 623, 180069, 86333, 60500, 220, 23]` is what both backends write,
`the_bounded_loop_is_the_unbounded_one_bit_for_bit`,
`a_query_row_walks_the_keys_its_window_and_its_position_leave_it`,
`a_calls_rows_share_a_weight_read_only_where_they_name_one_expert` and
`a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine` pass
unrelaxed, and both shape floors are where they were.

**What the discipline cost is one case and half the tier again.** The timing tier
gained one measurement — the streaming read — and is forty-eight where it was
forty-seven; the fast tier gained two ordinary cases holding `both_ways` to its
ordering. The three sweeps that were rewritten are the same three that were
already there, and running their arms twice after two seconds of warm-up is what
takes the tier to 19.9 minutes where it was about 13.

### What this milestone leaves

**The reference's arm is the thing to keep re-measuring, and now there is a
reason rather than a habit.** Four milestones declined to re-take it and each was
right about the wall times and the prefill; what none of them could have known is
the decode row above. **A column nobody re-reads is not stable, it is
unobserved** — and the cost of finding out was one sitting.

**Three things are named and sized rather than done.** The block's height as a
property of the compiled entry rather than a constant, which unlocks both the
taller block and the 64-to-128 thread width and is one piece of work; the
differential corpus large enough to bound a disagreement rate, which is the item
A7 named and no milestone since has done; and the matmul's accumulator chain
split, still four lines. **The first is the one this milestone would have done
had the premise it was given been right** — the floor was said to cost part of a
speculative round and it costs none of one.

**And the instrument grew a control it did not have.** A null pair through the
paired harness — the same build against itself, reading no claim — is what says
the harness is not inventing effects, and this file had never printed one.

## The tail of a step

**What a decode step did last was a round trip, and the only reason was that the
final norm ran on this side.** The stack's rows came back so this process could
normalise them, divide by `logits_mup_width_multiplier`, and hand the result
straight over again for `lm_head` — two crossings and a submission, around a
projection whose input the device had just written. A chain of eight MTP heads
paid the same thing eight times: half of its sixteen submissions were `lm_head`,
one behind each head.

**All three run there now**, in whichever command buffer wrote the rows they
read — the run of layers for a decode step, the head's own for a guess. Nothing
else moved: the last layer of the stack is simply a layer with something after
it, which is the condition every other merged submission in this file rests on.

**This is the one piece of tail work the project deferred every time it came up,
and the reason was numerical.** Apple GPUs have no f64, a norm is a reduction,
and a reduction that reassociates differently from the host moves a logit — and a
moved logit at the top of the distribution is a different token. So the
comparison came before any timing: the same prompt through two stacks differing
in nothing but who runs the last three operations, so that what they hand the
tail is the same hidden state bit for bit.

    step   token   logits apart   the winner's margin   normed apart
     0       656        0.000e0               1.449e-1        6.729e-8
     1        13        0.000e0               5.683e-2         0.000e0
     2       623       1.715e-7               1.404e-1        7.120e-8
     3    180069       1.780e-7               1.283e-1        6.904e-8
     4     86333        0.000e0               1.407e-1         0.000e0
     5     60500        0.000e0               1.484e-1         0.000e0
     6       220       1.468e-7               1.228e-1        1.424e-7
     7        23       1.794e-7               1.556e-2        6.303e-8
     8     33610        0.000e0               1.708e-1         0.000e0

**The same token at every one of the nine positions, and the margin column is
what says that is not luck.** The worst the two tails disagree by is 1.79e-7 of
the row's peak and the narrowest gap between a position's best logit and its
second best is 1.56e-2 — a factor of 87000, and the assertion the case makes is
that the first stays under the second. Five of the nine positions agree bit for
bit and four differ in the last couple of ulps, which is a reduction landing on
the same float most of the time and not quite always; the row at step 0 is the
one where a normed state that differs does not produce logits that do.

**The muP divide is exact rather than close, and it is exact because of where it
went.** A dispatch of its own for one multiply is a dispatch, and the norm's own
per-row scale is a *multiply* where the reference divides — so the multiplier is
divided into a copy of the norm's weight instead, once, at wrap time. Scaling by
a power of two moves an exponent and no mantissa bit, so `a * (w/m)` and
`(a * w)/m` are the same float; both halves of that are checked rather than
assumed, and a checkpoint whose multiplier is not a power of two, or whose
divided weight would fall subnormal, keeps the tail it always had. What the
round trip of the division would say is nothing — `0.3 / 12.0 * 12.0` is `0.3`
and `0.3 / 12.0` is not exact — which is a case the tests here name.

**What it is worth is a submission each time, and that is what it measures.** A
decode step is **1078 dispatches in 14 submissions** where it was 1077 in 15, and
the round-trip table says which one went. Its first two rows, as they read
immediately before this change:

    dispatches   a step     waited  scheduled    queued   executed  unattributed
    1                 1   969.00µs    60.34µs  134.66µs   660.17µs      113.83µs
    52                1   838.60µs    97.29µs   10.82ms   835.92µs        0.00ns

The first is `lm_head`'s own submission and there is no such row now; the second
is the last two layers of the run, and it is 54 dispatches with the tail behind
them. Over three readings that row is **0.89 to 1.01 ms of wait around 0.66 ms of
execution, so 0.23 to 0.35 ms of it bought nothing** — which is this milestone
re-measuring a figure the file had at 1.3 ms in the M16 era, against a step that
has changed twice since, rather than inheriting it.

**About 0.3 ms is also what the step moved.** Over three alternating pairs with the
order of the two halves flipped each pair, and every pair moving the same way, an
unspeculated decode step over 64 tokens went **21.18 ms to 20.86** — ranges
21.14-21.23 against 20.83-20.89 — and a run that keeps four timesteps of slack in
every window went 21.23 to 20.90 beside it. The two agree with the round-trip
row to a hundredth of a millisecond, which is what says the submission is the
whole of what was removed.

**A chain of eight heads is 168 dispatches in 8 submissions** where it was 160 in
16 — 21.0 a submission against 10.0, and against a decode step's 77. Over five
alternating pairs, every pair moving the same way and the two ranges not
overlapping, the chain went **27.36 ms to 25.31**, and the device's own clock did
not move with it: 18.21 ms against 18.26, the 0.05 being what eight more norm
dispatches cost at 6 µs each. So the 2.05 ms is round trip and nothing else, at
0.26 ms a submission removed — the same price the decode step paid for its
one.

**A prefill is where the fold does not reach, and the budget is why.** A run ends
when it has retained the bytes it may hold, and that is asked before the tail is:
385 and 769 tokens are 1117 dispatches in 42 and 43 submissions either way, with
the norm and the divide on this side and `lm_head` in a submission of its own,
exactly as before. 97 tokens is under the budget at its last layer and does fold
— 1118 dispatches in 21 submissions where it was 1117 in 22 — and it costs
nothing measurable: 1.21, 1.20 and 1.21 s of wall against 1.21, 1.23 and 1.21,
and 497, 499 and 494 ms of device time against 493, 499 and 498. Every other
column of that table is identical to the tenth at all three lengths.

**No token changed anywhere.** The recorded continuation is the recorded
continuation; 48 tokens of a longer prompt are byte for byte what they were at
`k` of 0, 1, 2 and 4; and acceptance is unmoved to the digit — 85% at depth 1,
then 91/74%, 84/74/63% and 82/65/53/47% — which is what says the guesses the
heads make are the same guesses. **The peak resident set did not move either**,
and the measurement it is worth reading is the command's own rather than a test
harness's: `inklingrs generate` over the same eight tokens peaks at 402.2, 402.3
and 402.4 MB before and 402.7, 402.8 and 403.0 after, over three pairs. The
gated case that bounds a *pass* reads 0.08 GiB against 0.22 across the same two
commits, and none of that is the engine: all of it is `Device::open`, 7 MiB in
one test binary and 144 MiB in the other, before a line of model code runs and
with the tail withheld. `device.rs` is byte-identical across the two, and running
each binary from the other's directory moves nothing — the cost follows the
binary. Why the driver charges one process twenty times what it charges another
for opening the same device is not attributed here.

**Sampling did not follow here and has since**, and what it had to buy first was
not a tolerance. The argmax over 200058 logits was the last thing a step asked
this process for — `sample` at 279 to 288 µs over three readings, 1.4% of a
decode step, beside a `readback` of 61 to 74 µs — and on a chain of eight it was
worth far more and measured far worse. What a device argmax has to reproduce is
the tie rule the whole engine's token identity rests on, `greedy` being
`top_k(logits, 1)` and taking the lower id, over a reduction across eighty cores.
That is its own equivalence argument and not a corollary of this one, so it has
its own section: see "Sampling on the device" below.

There is no operation of a layer left outside the GPU, and nothing of the model
past its last layer either. Both backends generate the same tokens, and the CPU
one stays the oracle every kernel here is validated against.

## Sampling on the device

**The argmax was the last thing past the model this process still did**, and it
is the only operation in a step whose whole answer is four bytes: `lm_head`
writes 200058 floats, they cross a seam, and a pass over them names one id. Two
milestones declined to move it and both were right to, for a reason that is not a
numerics one. **Every other kernel in this engine is held to the CPU within a
tolerance; this one has to be held to it exactly.**

**The tie rule survives a tree because of what the tree combines.** A candidate
is a value and the id holding it; combining two keeps the larger value and, where
the two agree, the lower id; and that operator is a *maximum under a total order*
— candidates ranked by value ascending and, within a value, by id descending. A
maximum over a set is the same element whatever order the set is folded in, so
the tie rule is not a property each combining step has to re-establish: it is a
property of the operator, and every step is the same operator — a thread's own
stripe, a simdgroup, a threadgroup, and the second dispatch over what the
threadgroups left. The empty candidate is that operator's identity, which is what
lets a stripe with nothing in it take part rather than be branched around.

**And nothing here compares floats.** `top_k` ranks with `total_cmp`, which is a
total order over every float: it separates `-0.0` from `0.0` and places a NaN at
whichever end its sign says. A kernel comparing with `>` agrees everywhere the
two values are distinct and disagrees exactly where a tie rule is what is being
tested — `-0.0 > 0.0` and `0.0 > -0.0` are both false, so a float comparison
calls them tied and takes the lower id where the host takes the `0.0`. So a
float's bits are mapped to the unsigned integer that ranks them and the reduction
compares integers, which makes the equality the whole rule turns on **bit
equality** and makes NaN and `-0.0` answers rather than caveats.

**The tie cases were written before the kernel, and a reduction that is correct
on random logits passes none of them.** A tie in each of the four places this
reduction breaks one — two lanes of a simdgroup, two simdgroups, one thread on
two of its own strides, and two threadgroups — and the cross-threadgroup one at
three different cuts, so that the second dispatch is asked to break it as two
lanes, as two simdgroups and as one thread twice. A tie at the padding edge. A
tie in a row of a block that is not the first. `-0.0` against `0.0`, NaN at both
ends, subnormals, infinities. And **two mutations of the rule**, each asserted to
move the ties it decides *and to leave the others where they are* — because a
mutation that moved every answer would mean the cases were pinned by something
coarser than the rule they are written for.

**966 slots that must never win a reduction.** `lm_head` is 201024 rows and
200058 of them are vocabulary. The projection is already cut there, so the padded
rows are never multiplied — and the argmax is told the vocabulary anyway and
ranks nothing past it, because a cut made in one place is a cut that stops being
made the day something else feeds this. The case that says so fills those slots
with infinities and asks, and a second one puts a tie across the boundary so that
the padded id loses for being padding rather than for being higher.

**Two dispatches, which is the opposite of what the norm decided**, and the
sweep is what says the arithmetic flipped. A norm measured a split across
threadgroups over a 4096-wide row and declined it at four microseconds of
encoding against six saved. A row of the vocabulary is fifty times that, and one
threadgroup is one core of eighty:

    threadgroups a row      1       2       8      32      80     128     512
    one row              284µs   135µs  40.6µs  16.0µs  10.9µs   9.8µs  12.0µs
    nine rows            268µs   141µs  39.6µs  17.6µs  16.7µs  19.7µs  45.4µs

**One threadgroup is 284 microseconds, which is this process's own argmax to
within its spread** — so a device argmax on one core would have bought exactly
nothing, and the whole of what this is worth is the cut. Eighty is the machine's
own core count and is never more than 11% off the best cut at any block width,
where the fitted best is 128 at one and two rows and 80 at four and nine.

### What it is worth

**Over seven alternating pairs against the commit before it, on the packed-head
checkpoint** — every depth moving the same way in every pair, and no two ranges
overlapping at any of them:

    k                      0      1      2      3      4
    before ms/token    20.90  18.00  17.61  18.23  21.61
    after  ms/token    20.59  17.16  16.48  16.83  19.74
    before speedup     1.000  1.161  1.187  1.146  0.967
    after  speedup     1.000  1.200  1.250  1.224  1.043
    device, before     19.47  15.87  14.97  15.03  17.54
    device, after      19.49  15.87  14.97  15.03  17.52
    accepted                  85%   87/78%  85/65/55%  82/65/53/47%

**Every depth pays now, which has never been true in this file.** `k = 4` is
1.043× where it was 0.967×, and the best is still `k = 2` at **1.250× and 16.48
ms/token, 60.7 tokens/s** against 17.61 and 56.8. The deeper the round, the more
this is worth — 1.5% at `k = 0` and 8.7% at `k = 4` — because a round of depth
`k` takes `k + 1` argmaxes on the chain and one on the block, and every one of
them was a pass over 200058 floats on this side.

**The split over the key span then moved every one of those absolute figures and
narrowed every ratio**, over seven alternating pairs against the commit before
it, on the same checkpoint:

    k                      0      1      2      3      4
    before ms/token    20.563 17.135 16.363 16.813 19.481
    after  ms/token    19.414 16.583 16.078 16.591 19.351
    before speedup     1.000  1.200  1.257  1.223  1.056
    after  speedup     1.000  1.171  1.207  1.170  1.003
    device, before     19.460 15.861 14.950 15.004 17.517
    device, after      18.600 15.428 14.680 14.874 17.437

**Every depth is cheaper a token and every speedup is smaller, and those are the
same fact.** `k = 0` fell by 5.6% — seven pairs of seven, ranges apart — and the
speculating depths by 1 to 3%, so the ratios narrowed because the baseline moved
further than the rounds did. `k = 4` is 1.003× where it was 1.056×, which is the
one claim above this table gives back: it is 130 µs faster a token and no longer
comfortably worth running. **Acceptance is identical to the digit at every depth
in the same sitting** — 84.8%, 87.0/78.3%, 85/65/55%, 82.4/64.7/52.9/47.1% — and
tokens a round are 1.829, 2.560, 2.909 and 3.368 either side, which is what says
no guess moved. The recorded continuation did not change, nor did any of the 583
gated cases.

Those figures are all taken at a 34-token prompt over 64 tokens, where a span of
98 keys is four tiles and the split is four. What the same change is worth at a
context somebody has is under "Where a decode step goes as the context grows",
and it is a great deal more.

**The device's own clock did not move at any depth**, which is the whole of what
says this is the asking rather than the work: 19.47 against 19.49, 15.87 against
15.87, 14.97 against 14.97, 15.03 against 15.03 and 17.54 against 17.52, with
every one of those five reading "no claim" by this file's own standard in the
same sitting the wall times read "ranges apart, seven pairs of seven". The two
argmax dispatches are 0.13 ms of a chain's device time and are the only thing
that was added.

**Acceptance is identical to the digit at every depth**, in the same sitting —
84.8% at `k = 1`, 87.0/78.3% at `k = 2`, 85/65/55% at `k = 3` and
82.4/64.7/52.9/47.1% at `k = 4`, banking 1.829, 2.560, 2.909 and 3.368 tokens a
round either side. Which is what it has to be: no guess moved, because the
argmax that makes a guess is the same argmax.

**Where the milliseconds came from is two rows of the profile and one of them is
the noisiest number in this file.** On a chain of eight heads, three warm
readings before put `sample` at 7.44, 5.12 and 2.38 ms — S3's own 2.2 to 7.2 ms,
tracking the drift of the runs around it rather than the work — and `readback` at
0.75, 0.69 and 0.52 ms. After, there is no `sample` row and `readback` is 0.091,
0.086 and 0.132 ms. The chain itself went **17.14, 17.87 and 17.99 ms to 14.64,
14.62 and 15.45**, and its device clock 10.18, 10.31 and 10.31 against 10.34,
10.34 and 10.38. On a decode step `sample` was 280 µs of 20 ms and `readback` 69,
71 and 73 µs; there is no `sample` row, and `readback` is 2.8, 3.5 and 4.9 µs.

**A prefill did not move and could not**, which is the control: 5591 ms against
5644 over seven pairs with the ranges across each other, and the device's own
clock 3813.55 against 3813.28. A prompt takes one token out of its last position
however many positions it had.

**The peak resident set went down rather than up**, over three pairs of
`inklingrs generate` at eight tokens: 402.8, 402.6 and 402.8 MB against 402.1,
398.8 and 398.6, and a run speculating two deep 431.5, 431.8 and 431.7 against
426.3, 426.3 and 394.5. The argmax's own allocations are a candidate a
threadgroup and one id a row — 644 bytes against the 800 KB of logits that stop
being copied into this process's memory once a step a token is all anybody wants.

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
that follow it — 20.9 ms, where the 20.1 ms above is the same step at the
eight-token context every other measurement in this file is taken at:

    tokens in the block    1      2      3      4      6      9
    forward pass       24.2ms 30.6ms 37.3ms 42.6ms 61.6ms  78.9ms
    × a decode step      1.16   1.47   1.79   2.04   2.95    3.78
    submissions            14     14     14     14     14      14

    heads chained          1      2      3      4      6      8
    the chain           3.6ms  6.6ms  9.5ms 12.7ms 19.1ms  25.3ms
    × a decode step      0.17   0.31   0.46   0.61   0.91    1.21

**The first two rows of the block table are noise and are worth saying so
about**: over three readings a block of one swings 19.9 to 26.1 ms and a block of
two 28.9 to 32.0, either side of the change these figures were taken across, so a
row that reads above a decode step at one token is the measurement rather than
the engine. From three tokens up the same three readings hold to a few tenths,
and hold to a few tenths across the change: 37.8 to 37.3, 42.2 to 42.6, 61.3 to
61.6 and 79.1 to 78.9 ms.

**An extra token in the block costs 6.8 ms**, which is 0.33 of a decode step and
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
a round can ask for. The fourteen in the row are command buffers a run commits as
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

A head's guess cost 3.2 ms and read 995 MB — its own 532 MiB, and `lm_head` again
to turn a hidden state into a token; its own half of that is 141 MiB now, for
the reason "Quantising the heads" below gives. **How that divides was inferred here
for two milestones, measured against the inference and found to disagree with
it, and has now been moved**: at six submissions a guess the device executed for
2.2 ms of 4.5 and the other 2.3 were the round trips and this process's own work
between them, where this file had 3.4 ms of bandwidth and 1.3 of round trip. At
one submission a head plus its `lm_head` it was 2.27 ms of execution inside 2.78
ms of wait, and at one submission for both it is 2.28 inside 2.58. The study called the reference's per-head overhead "yours to win in
Rust"; mlx-vlm was near the old figure at 3.9 ms — on the 8-bit checkpoint, whose
heads are these heads byte for byte but whose `lm_head`, which a guess also
reads, its quantiser left in the original precision. **Only the `lm_head` half
of those bytes was the packed matmul's** when that was measured, and reading four
to a lane took that dispatch 1.57 ms to 1.46 and the guess 4.85 to 4.70 — the
same tenth of a millisecond arriving twice, which is what said the head's own
bfloat16 tensors were the other half. Both halves are the packed matmul's now.

**And now where a chain's milliseconds go, which is a question this file had
answered for a decode step and a prefill and never once asked of the heads.**
The same tables, over the eight heads at one row, sampled — as they read before
the heads were packed, and as they read now:

    kernel            calls    device   share       moved   achieved  of peak
    dense_matmul         72   12.87ms   65.6%   4469.46 MB   347 GB/s     42%
    packed_matmul         8    5.75ms   29.3%   3489.14 MB   607 GB/s     74%
    rms_norm             40  495.23µs    2.5%      2.37 MB     5 GB/s      1%
    short_conv           32  320.42µs    1.6%      5.77 MB    18 GB/s      2%
    fused_attention       8  131.81µs    0.7%      1.05 MB     8 GB/s      1%
    swiglu                8   47.62µs    0.2%      3.15 MB    66 GB/s      8%

    packed_matmul        80    9.73ms   91.1%   5866.69 MB   603 GB/s     74%
    rms_norm             40  429.10µs    4.0%      2.37 MB     6 GB/s      1%
    short_conv           32  268.69µs    2.5%      5.77 MB    21 GB/s      3%
    fused_attention       8  118.52µs    1.1%      1.05 MB     9 GB/s      1%
    argmax                8   86.18µs    0.8%      6.41 MB    74 GB/s      9%
    swiglu                8   45.62µs    0.4%      3.15 MB    69 GB/s      8%
    argmax_combine        8   41.49µs    0.4%      0.01 MB     0 GB/s      0%

**The row that was two thirds of the table is not a smaller row, it is absent.**
`dense_matmul` was the only kernel in the model reading a format nobody packed,
and there is no such format in a chain any more: the eighty calls of the second
table are the seventy-two that were `dense_matmul`'s and the eight `lm_head`
already was. What the two tables are of is under "Quantising the heads" below.

**A chain of eight heads read 7.96 GB where a decode step reads 5.9**, and 4.5
of those were bfloat16 the quantisers never touched — so two thirds of the
chain's device time was the one kernel in the model reading a format nobody
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
the convolutions now carry as a second addend. The `rms_norm` row is forty calls
where it was thirty-two, and the eight are the model's own final norm arriving
here — see "The tail of a step".

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
nineteen in one command buffer, and one for the `lm_head` behind it. **And it is
168 dispatches in 8 now**, because the `lm_head` behind each head is in the
head's own buffer along with the final norm in front of it — 21.0 a submission,
against a decode step's 77. Of a 25.3 ms chain the device executes for 18.2 and
the wait is 20.7, where of the 43 ms chain two milestones ago the device executed
for 17.9 and the wait was 36.5:

    dispatches   a chain     waited  scheduled    queued   executed  unattributed
    21                 8    20.72ms   631.15µs  712.15µs    18.24ms        1.13ms

**`queued` is 0.71 ms across the whole chain where a decode step's run of layers
has 71 ms of it, and that is not what merging a head fixed.** M16's pipelining
still reaches none of this, and cannot: a head's guess has to *be* a token before
the head after it can embed it, so there is nothing behind the buffer being
waited for and there is no arrangement of these eight submissions in which
there would be. What the merge removed is the other 32 — the norms, the two
convolutions, the head norms, the activation and the residual adds each had a
kernel on the device already and what did not exist was the seam. It is the same
move the layers made four milestones ago, and what it needed was not a kernel:
`LayerProjections` and `DenseFfn` hold a weight either format answers for, so a
head's block is wrapped as the decoder layer it always was.

**What was left after those three was this process's own, and it was the argmax
and the logits arriving to be argmaxed** — eight times over, for the same reason
`queued` is nothing. `sample` was the noisiest row in this file while it existed:
five sampled readings put it between 2.2 and 7.2 ms, tracking the drift of the
runs around it rather than the work. It is not a row any more and `readback` is
0.5% of a sampled chain where it was 2.1%, both for the reason "Sampling on the
device" above gives — a head's guess comes back as an id and not as the 200058
logits it was taken from. A chain of eight is **14.6 to 15.5 ms** where those
readings were taken at 17.1 to 18.0, and its device clock did not follow.

**What was left of the chain was the format, and it is taken below.** Of 27.4 ms
the device executed 18.2, and 12.6 of those were `dense_matmul` reading the 4.5
GB of bfloat16 the quantisers never touched — so the largest thing left in a
chain was not a round trip at all but the format the MTP shard ships in. Of the
2.4 ms of wait that is not execution, the eight `lm_head` submissions used to be
half — and they are not there any more, because the model's own final norm is on
the device beside them.

## Quantising the heads

**The heads were the one part of this checkpoint nobody had ever quantised.**
Every quantiser dropped or skipped `model.mtp.*`, so the shard beside an MXFP4
stack is the BF16 original's 160 tensors byte for byte — and one kernel reading
those 4.5 GB was two thirds of a chain's device time. `just quantize-mtp` writes
them in the format the stack is already in.

**MXFP4 and not 8-bit, and the reason is which side the work is on.** `just
quantize` produces 8-bit affine and is what built `Inkling-Small-8bit`, so on
the *writing* side 8-bit is the format this repo already has. On the reading
side it has exactly one: `inkling_core::quant` is MXFP4 — E2M1, group 32, E8M0
scales — and so is every kernel that multiplies without decoding. 8-bit heads
would have been a second dequantiser, a second kernel and an equivalence
argument for both; MXFP4 heads are none of those, because the packed matmul
every projection of the stack goes through already reads them. What it cost
instead was the quantiser, and that turned out to be `mx.quantize(mode="mxfp4")`
— MLX's own, which is what produced the stack these heads sit beside, so the
codes written are the codes the dequantiser is pinned to rather than a second
implementation of the same table. **Eight tensors a head**: the norms, the
convolution kernels and the relative-position table are 260 KB a head against
532 MiB and no matmul reads them.

**It is a new quantisation and not a re-quantisation**, which is what makes the
gate below the whole of the milestone: these are original-precision weights, so
nothing about them has been through a quantiser before and there is no earlier
loss to compare against. 4.157 GiB of bfloat16 become **1.105 GiB of codes and
scales, 3.76×**, and the relative error the codes carry is 0.118 to 0.134 per
tensor with the worst at `input_proj` — a figure `--check` prints per tensor,
because what acceptance is at risk from is exactly that and a single tensor
taking it far worse than the rest is the thing to know before any of this runs.

**The bfloat16 shard is not touched and must not be.** It is the oracle the
packed heads' guesses are held against, and the two checkpoints are the same
140 GB stack read twice: `models/Inkling-Small-mxfp4-mtp4` is forty symlinks and
one shard of its own, so what this costs on disk is **1.10 GiB beside a 131 GB
checkpoint**. A loader maps every `*.safetensors` in a directory, which is why
the packed heads are a directory rather than a second shard beside the first.

**The two formats are one question per weight.** What says a weight is packed is
the `.scales` tensor beside it, asked per weight rather than per shard, and
`Multiply` is where the two meet on the device — so nothing a head is multiplied
*into* changed: the norms, the convolutions, the attention step and the command
buffer all nineteen dispatches share are the layer's own either way. The one
weight the two formats disagree about the shape of is the SwiGLU. A bfloat16
shard fuses its gate and its up interleaved row by row and the kernel reads
every other row; a packed pair cannot be strided through — codes, group
boundaries and scale bytes would all have to be — so a packed shard holds the
two apart, which changes nothing about either: a group spans 32 values of a row,
and which rows are in a tensor is not something quantisation can see.

### The gate is acceptance, and it is not the tokens

**No token can move.** The model verifies every guess, so a worse head produces
a rejected guess rather than a wrong output — which is real safety and is also
why the usual test is blind here. What is at risk is *acceptance*, and
acceptance is the whole of the speedup. So the two chains are held against each
other before any timing claim: one generation, one stack, one set of embeddings,
both chains asked the same round at every round of it, and the count of where
they answered differently reported by depth. `just guesses <a> <b>` is that —
the one measurement in this file that compares answers rather than durations —
and over 64 tokens of the structured prompt:

    depth                    1      2      3      4     all
    guesses                 18     18     18     17      71
    diverged                 0      3      3      4      10

**The first head guesses what the bfloat16 first head guesses, every time.**
That is where most of the acceptance is — 85% at depth 1 against 47% at depth 4
— and it is the depth that decides whether speculation pays at all. Of the
seventy-one guesses the two chains were both asked for, ten differ, all of them
past the first head. Driven the other way round — the packed chain proposing and
the bfloat16 one answering beside it — the same run reads 0, 3, 2 and 4 against
the same 18, 18, 18 and 17, so which of the two the generation belongs to moves
one guess.

**And acceptance itself barely moved**, measured in the same sitting as the
timings below, five alternating pairs with the order flipped each pair:

    k                         1        2           3              4
    bfloat16 heads          85%   91/74%   84/74/63%   82/65/53/47%
    packed heads            85%   87/78%   85/65/55%   82/65/53/47%
    tokens a round, bf16  1.829    2.560       3.048          3.368
    tokens a round, mxfp4 1.829    2.560       2.909          3.368

**Identical at `k = 1` and at `k = 4`, to the digit, at every depth of both.**
At `k = 2` the two depths trade — 91/74 against 87/78 — and bank exactly the
same 2.560 tokens a round, so what changed there is which of the two guesses was
the one rejected. **`k = 3` is the one that lost anything**: 3.048 tokens a round
to 2.909, 4.6% fewer, which over the same 64 tokens is twenty-two rounds where it
was twenty-one. That is the
trade, stated rather than left for the reader: at `k = 3` the packed chain is
8.8% cheaper a token and banks 4.6% fewer tokens a round, and the first of those
is larger than the second — which is why the depth still comes out ahead, at
1.131× against 1.034×.

### What the heads' format is worth

**Over five alternating pairs, one build against two checkpoints, the order
flipped each pair** — `just bench-weights models/Inkling-Small-mxfp4
models/Inkling-Small-mxfp4-mtp4 sweep`, which is one sitting and one stack with
the heads the only thing that differs:

    k                      0      1      2      3      4
    bfloat16 ms/token  20.95  19.01  19.86  20.28  25.19
    packed   ms/token  20.91  18.11  17.77  18.49  21.79
    bfloat16 speedup   1.000  1.102  1.055  1.034  0.832
    packed   speedup   1.000  1.155  1.177  1.131  0.960
    device, bfloat16   19.50  16.83  16.86  16.79  20.75
    device, packed     19.49  15.89  14.99  15.04  17.55

**Every depth that speculates moved, every pair moved the same way, and no two
ranges overlap** — 18.906-19.132 against 17.910-18.624 at `k = 1`,
19.498-20.077 against 17.362-18.270 at `k = 2`, 19.578-20.671 against
18.229-18.892 at `k = 3` and 24.485-26.051 against 20.957-22.823 at `k = 4`.
**And `k = 0` did not**, in the same sitting: 20.953 against 20.907 with the
ranges lying across each other and three of the five pairs falling the other
way, which is the control this comparison has and the reason it was run at a
depth that maps no head at all.

**`k = 2` is what pays best now and it pays 1.18×**, where before it was `k = 1`
at 1.10×. Three depths pay more than the best depth used to, `k = 3` has gone
from 1.034× to 1.131×, and **`k = 4` is 0.96× where it was 0.83×** — still a
loss, and now a loss by four percent rather than by seventeen. The whole shape of
the curve changed: what a round pays to guess fell by enough that the depth worth
running moved outward, which is the first time in this file that has happened.

**The device's own clock moved with it and by more**, which is what says this is
the weights rather than the scheduling: 16.86 ms to 14.99 at `k = 2` and 20.75 to
17.55 at `k = 4`. A chain of eight heads over one row is **25.21 ms to 17.28**,
each sitting's own, and its device time **18.92 ms to 10.26** — 8.7 ms of
execution off a chain, where `dense_matmul` was charged 12.9. **The time fell by
more than the bytes did**, 1.84× against the 1.35× the declared column reads
(7.96 GB to 5.88), and the rates are why: the kernel that replaced the bfloat16
one runs at 599 GB/s where it reached 347. A chain's `lm_head` is 3.5 GB of those
reads, was packed all along, and did not move at all.

**A round saves more than the chain-over-one-row table can explain**, and this
file has the same shape of finding on record. At `k = 2` a round banks 2.560
tokens and costs 50.8 ms against 45.5, which is 5.35 ms; the chain of two heads
over one row is 6.33 ms against 4.41, which is 1.92. A proposer runs its heads
over *every* row the round committed rather than over the last one alone — see
`Round`'s own paragraph — so a chain timed over one row is a lower bound on what
a round pays for it, and the factor between the two figures is about the rows.
That is the same "a block timed against a warm cache and the round a generation
pays are not the same measurement" the tables above already carry, met from the
other side.

**The free-chain ceiling this file has quoted since S1 cannot be quoted here, and
that is a finding rather than an omission.** It is arrived at by taking the
chain-over-one-row figure off a depth's ms/token, and doing that on both sides of
this sitting puts the floor at 16.7 and 16.9 ms/token with the bfloat16 heads and
at 15.7 and 15.4 with the packed ones. **A free chain has to leave the same floor
either side** — what it is a floor *of* is the block a round verifies, and the
block is the same forward pass through the same stack — so the 1.3 ms between
those two is the one-row figure being wrong about what a round pays for its
chain, by the amount the paragraph above predicts. No ceiling in this section is
stated from it.

**And nothing else about a round changed.** Tokens a round are identical at three
of the four depths, the block a round verifies is the same forward pass through
the same stack, and the recorded continuation is the recorded continuation: the
whole gated suite passes against the packed-head checkpoint, which includes the
cases asserting that 48 tokens of a longer prompt are byte for byte what they are
at `k` of 0, 1, 2 and 4, and that `--backend cpu` answers what it answered. **The
peak resident set did not move and would not**: neither format is copied onto the
device, so `inklingrs generate` over eight tokens peaks at 402.6-403.1 MB with the
bfloat16 heads and 402.8-405.7 with the packed ones over three pairs, and a run
speculating two deep at 422.2-422.3 against 422.8-425.2.

**And the code that reads either format cost the unspeculated path nothing**,
which is the other half of the same claim and a different comparison: two builds
against the one bfloat16 checkpoint, seven alternating pairs. A decode step is
20.952 ms against 20.906 with the ranges lying across each other and four of the
seven pairs falling one way, and the device's own clock 19.530 against 19.498 the
same way — no claim by this file's own standard, which is what a change that maps
no head at `k = 0` should read as. A 385-token prefill is 3.19 s against 3.30
over three pairs, ranges across; its device time is 1.8477 s against 1.8482,
which the same rule calls a claim over 0.03% and which is nothing — **the two
tests are necessary and how large a difference is remains the reader's**, and at
three pairs on a figure this repeatable the ranges will sit apart over a
rounding.

**Against mlx-vlm, the cross-engine table is now its own section** — see
"Against the reference, end to end", which is `just bench-engines` and supersedes
the hand-run sitting this paragraph used to hold. What that sitting said, at a
27-token prompt and 128 tokens out, was 22.66 ms a token for the reference
against 22.07 for this engine unspeculated and 26.87 at `k = 2`, and the last of
those three is the one worth carrying forward: **`k = 2` was a loss on that
prompt**, worth 0.82× against this engine's own unspeculated step where the sweep
prompt puts it at 1.25×. Nothing about the speculation changed between the two —
the same 128 tokens come out at every depth — and what differs is the text: that
prompt's first head is accepted 66% of the time against the sweep prompt's 85%.
**Acceptance is the workload's and the depth worth running is the workload's**,
which is the same finding the study's spread across regimes is and is why
`--speculate` takes a number.

The reference's own figures were re-measured in the new sitting and had not
moved. Swap was at zero and free memory at 310 GiB when it opened, the GPU was
idle before the first pair, and the four vllm-mlx daemons
`reference/results/prefill.md` already counts were resident between them. Two
things an earlier sitting recorded about the reference this one had no reason to
disturb: two of its twelve runs prefilled their own 27-token prompt at 27.8 tok/s
against 196–202 for the other ten, a 7× swing inside the process that never
reached its decode rate; and its model load was 6.5–7.1 s while its pages were in
the buffer cache and 20.7 s once the 8-bit checkpoint had evicted them.

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
quantisation. **They were not a quantisation of anything.** Every quant that kept
them kept them in bfloat16, and the 8-bit quant's `mtp.safetensors` is the BF16
original's own 160 tensors *byte for byte*, all 4.5 GB of them compared. So the
heads pair with any stack quantised from the same original, and giving this one
its heads is `just mtp-shard` — a file copy, where re-quantising the 532 GB
original to keep them is hours and would write out these same bytes. What it
costs is that the heads see an mxfp4 stack's hidden states rather than the
8-bit stack the acceptance study measured, which is why acceptance is measured
here again rather than inherited.

**They can be one now, and `just quantize-mtp` is what makes them one**: the same
heads packed MXFP4 into a checkpoint of forty symlinks and a 1.10 GiB shard of
its own, which is a *new* quantisation of original-precision weights rather than
a conversion of an existing one. Both checkpoints are kept and both load — what
says which format a weight is in is the `.scales` tensor beside it — because the
bfloat16 shard is the oracle the packed heads' guesses are held against. See
"Quantising the heads" above for what that is worth and what it cost.

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
