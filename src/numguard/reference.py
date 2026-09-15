"""Independent high-precision reference implementations.

These are deliberately NOT built from the same numpy/float code paths as
the naive/stable kernels under test in `kernels.py` -- they use Python's
arbitrary-precision `decimal.Decimal` at 50 significant digits, computed
straight from the mathematical definitions. This gives an independent
ground truth: a bug shared between the "naive" and "stable" numpy
formulas (e.g. both computing the wrong quantity) would not be masked by
comparing them only to each other.

Decimal's precision (50 digits) is vastly higher than float64 (~15-17
significant digits), so rounding error in these reference functions is
negligible relative to the errors we are trying to measure in float16/
float32/float64 kernels.
"""
from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Sequence

_PREC = 50


def _ctx():
    ctx = getcontext().copy()
    ctx.prec = _PREC
    return ctx


def _decimal_cos(x: Decimal, ctx) -> Decimal:
    """cos(x) via Taylor series after reducing x modulo 2*pi, computed
    entirely in Decimal arithmetic at the reference context's precision.

    Deliberately NOT built from math.cos (a float64 function) -- using a
    float64 trig call as the "ground truth" for a tool whose whole job is
    auditing float16/float32/float64 kernels would make the reference no
    more precise than the float64 kernel under test at exactly the
    dtype where the RoPE bug's residual error is smallest. This keeps
    the same independence guarantee as every other gold_* function here:
    a bug shared between the reference and the code under test cannot
    hide behind comparing them to each other.
    """
    pi = ctx.create_decimal(
        "3.14159265358979323846264338327950288419716939937510582097494"
    )
    two_pi = ctx.multiply(pi, ctx.create_decimal(2))
    quotient = ctx.divide(x, two_pi)
    k = quotient.to_integral_value(rounding=ROUND_HALF_EVEN, context=ctx)
    reduced = ctx.subtract(x, ctx.multiply(k, two_pi))
    x2 = ctx.multiply(reduced, reduced)
    neg_x2 = ctx.minus(x2)
    term = ctx.create_decimal(1)
    total = ctx.create_decimal(1)
    threshold = ctx.create_decimal(1).scaleb(-(_PREC + 5), ctx)
    for i in range(1, 60):
        denom = ctx.create_decimal((2 * i - 1) * (2 * i))
        term = ctx.divide(ctx.multiply(term, neg_x2), denom)
        total = ctx.add(total, term)
        if abs(term) < threshold:
            break
    return total


def _to_decimals(values: Sequence[float]) -> list:
    ctx = _ctx()
    return [ctx.create_decimal(repr(float(v))) for v in values]


def gold_logsumexp(values: Sequence[float]) -> Decimal:
    """log(sum(exp(x))) computed at 50-digit precision, shift-free.

    Decimal has no practical overflow limit for this input scale, so no
    shift trick is needed here -- this is the textbook definition,
    evaluated exactly enough to serve as ground truth.
    """
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, ctx.exp(x))
    return ctx.ln(total)


def gold_softmax(values: Sequence[float]) -> list:
    """softmax(x) as a list of Decimals, from the textbook definition."""
    ctx = _ctx()
    xs = _to_decimals(values)
    exps = [ctx.exp(x) for x in xs]
    total = ctx.create_decimal(0)
    for e in exps:
        total = ctx.add(total, e)
    return [ctx.divide(e, total) for e in exps]


def gold_cross_entropy(values: Sequence[float], target_index: int) -> Decimal:
    """-log(softmax(x)[target]) == logsumexp(x) - x[target], exact-ish."""
    ctx = _ctx()
    lse = gold_logsumexp(values)
    xt = _to_decimals(values)[target_index]
    return ctx.subtract(lse, xt)


def gold_mean(values: Sequence[float]) -> Decimal:
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, x)
    return ctx.divide(total, ctx.create_decimal(len(xs)))


def gold_variance(values: Sequence[float]) -> Decimal:
    """Population variance mean((x - mean(x))^2), the numerically safe
    two-pass definition, at 50-digit precision."""
    ctx = _ctx()
    xs = _to_decimals(values)
    mean = gold_mean(values)
    total = ctx.create_decimal(0)
    for x in xs:
        d = ctx.subtract(x, mean)
        total = ctx.add(total, ctx.multiply(d, d))
    return ctx.divide(total, ctx.create_decimal(len(xs)))


def gold_sum(values: Sequence[float]) -> Decimal:
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, x)
    return total


def gold_pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> Decimal:
    """Pearson's r = sum((x-mean_x)*(y-mean_y)) / sqrt(sum((x-mean_x)^2) *
    sum((y-mean_y)^2)), from the textbook definition, at 50-digit
    precision. Exactly-zero variance in either input is mathematically
    undefined (not a limit of finite values, an outright 0/0), matching
    scipy.stats.pearsonr's PearsonRConstantInputWarning convention of
    returning NaN rather than an implementation-dependent +-1."""
    ctx = _ctx()
    xs = _to_decimals(x_values)
    ys = _to_decimals(y_values)
    n = ctx.create_decimal(len(xs))
    mx = ctx.divide(sum(xs, ctx.create_decimal(0)), n)
    my = ctx.divide(sum(ys, ctx.create_decimal(0)), n)
    num = ctx.create_decimal(0)
    denx = ctx.create_decimal(0)
    deny = ctx.create_decimal(0)
    for x, y in zip(xs, ys):
        dx = ctx.subtract(x, mx)
        dy = ctx.subtract(y, my)
        num = ctx.add(num, ctx.multiply(dx, dy))
        denx = ctx.add(denx, ctx.multiply(dx, dx))
        deny = ctx.add(deny, ctx.multiply(dy, dy))
    if denx == 0 or deny == 0:
        return Decimal("NaN")
    den = ctx.sqrt(ctx.multiply(denx, deny))
    return ctx.divide(num, den)


def gold_layer_norm(values: Sequence[float], eps: Decimal) -> list:
    """(x - mean) / sqrt(variance + eps), from the textbook definition at
    50-digit precision -- variance here is always the mean-centered
    (numerically safe) population variance, since that is the
    mathematically correct quantity regardless of which formula a naive
    implementation uses to *approximate* it."""
    ctx = _ctx()
    xs = _to_decimals(values)
    mean = gold_mean(values)
    var = gold_variance(values)
    denom = (var + eps).sqrt(ctx)
    return [ctx.divide(ctx.subtract(x, mean), denom) for x in xs]


def gold_rms_norm(values: Sequence[float], eps: Decimal) -> list:
    """x / sqrt(mean(x^2) + eps), from the textbook RMSNorm definition
    at 50-digit precision -- unlike gold_layer_norm this is NOT mean-
    centered (RMSNorm deliberately omits mean-subtraction, that is the
    whole point of the "RMS" simplification versus LayerNorm), so the
    quantity under the square root is the raw mean of squares, exactly
    as both the naive and stable numpy kernels intend to approximate --
    the difference under test is only *which dtype the reduction runs
    in*, not the mathematical formula itself."""
    ctx = _ctx()
    xs = _to_decimals(values)
    total_sq = ctx.create_decimal(0)
    for x in xs:
        total_sq = ctx.add(total_sq, ctx.multiply(x, x))
    ms = ctx.divide(total_sq, ctx.create_decimal(len(xs)))
    denom = (ms + eps).sqrt(ctx)
    return [ctx.divide(x, denom) for x in xs]


def gold_kl_divergence(p_values: Sequence[float], q_values: Sequence[float]) -> Decimal:
    """sum(p_i * log(p_i / q_i)) at 50-digit precision, from the
    measure-theoretic definition with the standard 0*log(0/q) := 0
    convention applied explicitly (Decimal has no floating-point
    0*-inf indeterminate-form trap to fall into, but the *mathematical*
    convention still needs to be applied by this reference too, since
    Decimal(0) * Decimal(0).ln() would itself raise -- there is no
    "compute it and see what falls out" shortcut here; the convention
    is inherent to the definition of KL divergence, not an artifact of
    either implementation under test)."""
    ctx = _ctx()
    ps = _to_decimals(p_values)
    qs = _to_decimals(q_values)
    total = ctx.create_decimal(0)
    for p, q in zip(ps, qs):
        if p == 0:
            continue  # contributes exactly 0 by convention
        if q <= 0:
            return Decimal("Infinity")
        term = ctx.multiply(p, ctx.ln(ctx.divide(p, q)))
        total = ctx.add(total, term)
    return total


def gold_online_softmax(values: Sequence[float]) -> list:
    """The mathematical definition of softmax does not depend on how an
    implementation chunks its input -- chunking is purely an
    implementation strategy (e.g. FlashAttention-style tiling), not part
    of what the function means. So the ground truth for online_softmax
    is identical to gold_softmax: this is deliberate, and is exactly
    what lets naive_online_softmax's chunking bug be measured as a real
    error rather than an intentional difference in output.
    """
    return gold_softmax(values)


def gold_rope_cos(positions: Sequence[float], freq: float) -> list:
    """cos(position * freq) at 50-digit precision for every position, from
    the textbook RoPE angle definition -- the naive/stable kernels under
    test differ only in *what dtype the position*freq product and its
    cos() are computed in*, never in the formula itself, so this
    reference (built on the from-scratch Decimal `_decimal_cos` above,
    not a float64 trig call) is the correct independent ground truth."""
    ctx = _ctx()
    freq_d = ctx.create_decimal(repr(float(freq)))
    xs = _to_decimals(positions)
    return [_decimal_cos(ctx.multiply(x, freq_d), ctx) for x in xs]


def gold_masked_softmax(values: Sequence[float], mask: Sequence[bool]) -> list:
    """softmax restricted to the unmasked positions, at 50-digit
    precision, from the textbook definition: masked positions get
    exactly 0 probability, and the remaining positions get their
    ordinary softmax renormalized over just that subset. When every
    position is masked there is no valid probability distribution over
    an empty support -- by the same convention the stable kernel
    implements, this reference returns all zeros rather than raising,
    so the masked_softmax kernels can be scored like every other
    array-valued kernel instead of needing a special case in core.py.
    """
    ctx = _ctx()
    xs = _to_decimals(values)
    keep = list(mask)
    if not any(keep):
        return [ctx.create_decimal(0) for _ in xs]
    kept_xs = [x for x, k in zip(xs, keep) if k]
    kept_exps = [ctx.exp(x) for x in kept_xs]
    total = ctx.create_decimal(0)
    for e in kept_exps:
        total = ctx.add(total, e)
    result = []
    it = iter(kept_exps)
    for k in keep:
        if k:
            result.append(ctx.divide(next(it), total))
        else:
            result.append(ctx.create_decimal(0))
    return result


def gold_int8_add(
    a_code: float,
    b_code: float,
    zp_a: int,
    scale_a: float,
    zp_b: int,
    scale_b: float,
    zp_out: int,
    scale_out: float,
) -> Decimal:
    """Spec-correct affine-quantized element-wise add, at 50-digit
    Decimal precision: dequantize each operand with its OWN (scale,
    zero_point) -- real = (code - zero_point) * scale, the textbook
    affine/"zero-point" quantization definition shared by TFLite, ONNX
    QuantizeLinear/DequantizeLinear, and OpenVINO's LPT -- sum the two
    real values exactly, requantize into the output's int8 space, and
    saturate (clamp) to the representable range [-128, 127] rather than
    wrapping. This is deliberately independent of both kernel
    implementations under test (naive_int8_add/stable_int8_add in
    kernels.py): it is built directly from the quantization spec's
    algebraic definition, not from re-running either candidate
    implementation, so a bug shared between naive and stable could not
    hide behind comparing them only to each other.
    """
    ctx = _ctx()
    a = ctx.create_decimal(repr(float(a_code)))
    b = ctx.create_decimal(repr(float(b_code)))
    za = ctx.create_decimal(zp_a)
    zb = ctx.create_decimal(zp_b)
    zo = ctx.create_decimal(zp_out)
    sa = ctx.create_decimal(repr(float(scale_a)))
    sb = ctx.create_decimal(repr(float(scale_b)))
    so = ctx.create_decimal(repr(float(scale_out)))
    real_a = ctx.multiply(ctx.subtract(a, za), sa)
    real_b = ctx.multiply(ctx.subtract(b, zb), sb)
    real_sum = ctx.add(real_a, real_b)
    raw_code = ctx.add(ctx.divide(real_sum, so), zo).to_integral_value(
        rounding=ROUND_HALF_EVEN, context=ctx
    )
    qmin, qmax = ctx.create_decimal(-128), ctx.create_decimal(127)
    if raw_code < qmin:
        return qmin
    if raw_code > qmax:
        return qmax
    return raw_code


def gold_hll_register_term(rank: int) -> Decimal:
    """Spec-correct HyperLogLog register harmonic-sum term: exactly
    2**-rank, computed via Decimal exponentiation -- deliberately not
    built from any integer-shift code path (the thing under test),
    matching this file's independence guarantee that a bug shared
    between naive and stable could not hide behind comparing them only
    to each other."""
    ctx = _ctx()
    two = ctx.create_decimal(2)
    return ctx.power(two, ctx.create_decimal(-rank))


def gold_focal_loss_grad(logit: float, target: int, gamma: float, alpha: float) -> Decimal:
    """d(FocalLoss)/d(logit) via a symmetric central-difference numerical
    derivative of the focal loss ITSELF (not of either naive/stable
    analytic gradient formula under test), all computed in 50-digit
    Decimal arithmetic -- deliberately independent of both kernel
    implementations' analytic differentiation, so a shared algebra
    mistake in naive_focal_loss_grad/stable_focal_loss_grad could not
    hide behind comparing them only to each other.

    FocalLoss(x) = -alpha_t * (1 - p_t)**gamma * ln(p_t), where
    p_t = sigmoid(x) if target==1 else 1 - sigmoid(x), and alpha_t is
    alpha if target==1 else (1 - alpha) -- the textbook definition from
    Lin et al. (arXiv:1708.02002), evaluated directly (no chain-rule
    algebra applied by this reference at all). The step h=1e-20 is far
    below Decimal's 50-digit precision floor for the magnitudes this
    kernel's fixtures use, and well above the ~1e-50 rounding noise
    floor, giving a numerical derivative accurate to effectively all
    50 digits for smooth inputs (finite loss with p_t bounded away from
    a hard 0/1 in exact Decimal arithmetic, which never underflows the
    way float16/32/64 do -- that underflow is exactly the bug this
    kernel exists to catch in the kernels under test, not in this
    reference)."""
    ctx = _ctx()

    def sigmoid(x: Decimal) -> Decimal:
        return ctx.divide(Decimal(1), ctx.add(Decimal(1), ctx.exp(ctx.minus(x))))

    def focal_loss(x: Decimal) -> Decimal:
        p = sigmoid(x)
        if target == 1:
            pt = p
            alpha_t = ctx.create_decimal(repr(float(alpha)))
        else:
            pt = ctx.subtract(Decimal(1), p)
            alpha_t = ctx.subtract(Decimal(1), ctx.create_decimal(repr(float(alpha))))
        one_minus_pt = ctx.subtract(Decimal(1), pt)
        gamma_d = ctx.create_decimal(repr(float(gamma)))
        if one_minus_pt == 0:
            modulator = Decimal(0) if gamma_d > 0 else Decimal(1)
        else:
            modulator = ctx.power(one_minus_pt, gamma_d)
        return ctx.minus(ctx.multiply(alpha_t, ctx.multiply(modulator, ctx.ln(pt))))

    x = ctx.create_decimal(repr(float(logit)))
    h = ctx.create_decimal("1e-20")
    f_plus = focal_loss(ctx.add(x, h))
    f_minus = focal_loss(ctx.subtract(x, h))
    return ctx.divide(ctx.subtract(f_plus, f_minus), ctx.multiply(Decimal(2), h))


def gold_weighted_sampling_key(u: float, weight: float) -> Decimal:
    """log(u**(1/weight)) computed algebraically as (1/weight)*ln(u) in
    50-digit Decimal arithmetic -- mathematically identical to both
    naive_weighted_sampling_key and stable_weighted_sampling_key (they
    differ only in floating-point evaluation ORDER, not in the formula),
    so this reference is independent of which intermediate a kernel
    under test chooses to materialize."""
    ctx = _ctx()
    u_d = ctx.create_decimal(repr(float(u)))
    w_d = ctx.create_decimal(repr(float(weight)))
    return ctx.divide(ctx.ln(u_d), w_d)


def gold_weight_decay(values, q_values, num_steps: int) -> Decimal:
    """w0 * (1 - decay_per_step)**num_steps, computed exactly via
    Decimal exponentiation of the exact per-step multiplier -- the
    closed-form geometric-decay identity that AdamW's per-step decoupled
    weight decay is mathematically equivalent to (with no gradient
    term, matching this kernel's isolated decay-only model). This is
    NOT built by looping the kernel's own step-by-step multiply, so a
    bug shared between the naive and stable per-step loops (e.g. both
    using the wrong sign) could not hide behind agreeing with each
    other only.
    """
    ctx = _ctx()
    (w0,) = values
    (decay_per_step,) = q_values
    w0_d = ctx.create_decimal(repr(float(w0)))
    factor = ctx.subtract(Decimal(1), ctx.create_decimal(repr(float(decay_per_step))))
    return ctx.multiply(w0_d, ctx.power(factor, Decimal(num_steps)))


def gold_p2_quantile(values: Sequence[float], prob: float) -> Decimal:
    """P^2 (Piecewise-Parabolic) streaming quantile estimator (Jain &
    Chlamtac, CACM 1985), computed entirely in 50-digit Decimal
    arithmetic using the "recompute each marker's desired position from
    `count` every step" form. In exact (infinite-precision) arithmetic
    this is mathematically identical to the "accumulate a per-step
    increment" form naive_p2_quantile uses -- after k steps,
    k * dns[i] == count * p_i exactly -- so this single reference is
    independent ground truth for both naive_p2_quantile (accumulated
    `ns[i] += dns[i]`, which drifts under float rounding -- see
    AndreyAkinshin/perfolizer#8) and stable_p2_quantile (recomputed
    fresh each step, the author's own published fix). A bug shared by
    both float kernels (e.g. a wrong marker-selection or interpolation
    rule) would not be masked by comparing them only to each other,
    since this reference is built from the paper's formulas directly in
    Decimal, not from either kernel's float code path."""
    ctx = _ctx()
    p = ctx.create_decimal(repr(float(prob)))
    two = ctx.create_decimal(2)
    four = ctx.create_decimal(4)
    one = ctx.create_decimal(1)
    xs = [ctx.create_decimal(repr(float(v))) for v in values]

    q: list = [ctx.create_decimal(0)] * 5
    n = [0, 0, 0, 0, 0]
    ns: list = [ctx.create_decimal(0)] * 5
    count = 0

    def parabolic(i: int, d: int) -> Decimal:
        d_dec = ctx.create_decimal(d)
        denom = ctx.create_decimal(n[i + 1] - n[i - 1])
        term_a = ctx.divide(
            ctx.multiply(ctx.create_decimal(n[i] - n[i - 1] + d), ctx.subtract(q[i + 1], q[i])),
            ctx.create_decimal(n[i + 1] - n[i]),
        )
        term_b = ctx.divide(
            ctx.multiply(ctx.create_decimal(n[i + 1] - n[i] - d), ctx.subtract(q[i], q[i - 1])),
            ctx.create_decimal(n[i] - n[i - 1]),
        )
        return ctx.add(q[i], ctx.multiply(ctx.divide(d_dec, denom), ctx.add(term_a, term_b)))

    def linear(i: int, d: int) -> Decimal:
        return ctx.add(
            q[i],
            ctx.multiply(
                ctx.create_decimal(d),
                ctx.divide(ctx.subtract(q[i + d], q[i]), ctx.create_decimal(n[i + d] - n[i])),
            ),
        )

    for x in xs:
        if count < 5:
            q[count] = x
            count += 1
            if count == 5:
                q = sorted(q)
                n = [0, 1, 2, 3, 4]
                ns[0] = ctx.create_decimal(0)
                ns[1] = ctx.multiply(two, p)
                ns[2] = ctx.multiply(four, p)
                ns[3] = ctx.add(two, ctx.multiply(two, p))
                ns[4] = ctx.create_decimal(4)
            continue
        if x < q[0]:
            q[0] = x
            k = 0
        elif x < q[1]:
            k = 0
        elif x < q[2]:
            k = 1
        elif x < q[3]:
            k = 2
        elif x < q[4]:
            k = 3
        else:
            q[4] = x
            k = 3
        for i in range(k + 1, 5):
            n[i] += 1
        cnt = ctx.create_decimal(count)
        ns[1] = ctx.divide(ctx.multiply(cnt, p), two)
        ns[2] = ctx.multiply(cnt, p)
        ns[3] = ctx.divide(ctx.multiply(cnt, ctx.add(one, p)), two)
        ns[4] = cnt
        for i in range(1, 4):
            d = ctx.subtract(ns[i], ctx.create_decimal(n[i]))
            if (d >= one and (n[i + 1] - n[i]) > 1) or (d <= -one and (n[i - 1] - n[i]) < -1):
                d_int = 1 if d > 0 else -1
                qs = parabolic(i, d_int)
                if q[i - 1] < qs < q[i + 1]:
                    q[i] = qs
                else:
                    q[i] = linear(i, d_int)
                n[i] += d_int
        count += 1

    if count <= 5:
        ordered = sorted(q[:count])
        idx = int(round((count - 1) * float(p)))
        return ordered[idx]
    return q[2]


def gold_geometric_mean(values: Sequence[float]) -> Decimal:
    """(prod(x))**(1/n) computed algebraically as exp(mean(ln(x))) in
    50-digit Decimal arithmetic -- mathematically identical to both
    naive_geometric_mean and stable_geometric_mean (they differ only in
    floating-point evaluation ORDER: raw running product vs log-space
    accumulation, not in the underlying formula), so this reference is
    independent of which intermediate a kernel under test chooses to
    materialize. Decimal's 50-digit precision and unbounded exponent
    range mean this reference never overflows/underflows the way the
    naive float16/32/64 kernel does, giving a true ground truth for
    exactly the input ranges that break the naive formula."""
    ctx = _ctx()
    xs = _to_decimals(values)
    total_log = ctx.create_decimal(0)
    for x in xs:
        total_log = ctx.add(total_log, ctx.ln(x))
    mean_log = ctx.divide(total_log, ctx.create_decimal(len(xs)))
    return ctx.exp(mean_log)


def gold_repetition_penalty(values: Sequence[float], seen_mask: Sequence[bool], theta) -> list:
    """The MATHEMATICALLY CORRECT (gauge-invariant) answer against which
    both naive_repetition_penalty and stable_repetition_penalty are
    scored: apply the documented sign-branch penalty to Decimal log-
    probabilities computed from the exact input logits, at 50-digit
    precision. Because softmax (and therefore the true log-
    probabilities) are exactly shift-invariant, this reference produces
    the identical output distribution regardless of what additive
    constant the caller's raw logits happen to sit at -- which is the
    entire property naive_repetition_penalty violates and
    stable_repetition_penalty preserves. Deliberately NOT built by
    calling this repo's own naive/stable numpy code (see module
    docstring): an independent Decimal computation of the same
    normalize-then-penalize formula that stable_repetition_penalty
    uses, so a shared bug between the kernel under test and this
    reference cannot hide behind comparing them only to each other."""
    ctx = _ctx()
    xs = _to_decimals(values)
    m = xs[0]
    for x in xs[1:]:
        if x > m:
            m = x
    total = ctx.create_decimal(0)
    exps = []
    for x in xs:
        e = ctx.exp(ctx.subtract(x, m))
        exps.append(e)
        total = ctx.add(total, e)
    logz = ctx.add(m, ctx.ln(total))
    logp = [ctx.subtract(x, logz) for x in xs]

    theta_d = ctx.create_decimal(repr(theta))
    penalized = []
    for lp, seen in zip(logp, seen_mask):
        if seen:
            if lp > 0:
                penalized.append(ctx.divide(lp, theta_d))
            else:
                penalized.append(ctx.multiply(lp, theta_d))
        else:
            penalized.append(lp)

    m2 = penalized[0]
    for p in penalized[1:]:
        if p > m2:
            m2 = p
    exps2 = [ctx.exp(ctx.subtract(p, m2)) for p in penalized]
    total2 = ctx.create_decimal(0)
    for e in exps2:
        total2 = ctx.add(total2, e)
    return [ctx.divide(e, total2) for e in exps2]


def gold_speculative_reject(values: Sequence[float], q_values: Sequence[float]) -> list:
    """The MATHEMATICALLY CORRECT output distribution of one speculative-
    decoding rejection-sampling step: the Leviathan/Chen (2023) theorem
    guarantees that whenever the draft token is sampled from EXACTLY the
    same distribution p used in the accept/reject math (a_y=min(1,q_y/p_y),
    resampling from normalize(max(0,q-p)) on rejection), the resulting
    output distribution equals the target distribution q exactly -- this
    is the entire "lossless" guarantee that makes speculative decoding
    safe to deploy. This reference returns the 50-digit-Decimal softmax
    of the target logits directly, per that theorem -- deliberately NOT
    built by replaying either kernel's own r/p/q construction or its
    accept-reject arithmetic (see module docstring): a bug shared
    between the kernel under test and this reference (e.g. a wrong
    resampling formula) could not hide behind comparing them only to
    each other, since the correct answer here is derived purely from
    the theorem's statement, independent of any particular r/p
    articulation.

    `values` holds the draft logits (unused by this reference -- the
    theorem's guarantee does not depend on what the draft model
    proposed, only on q), `q_values` holds the target logits, matching
    every other two-distribution kernel's naming convention in this
    file (e.g. gold_kl_divergence).
    """
    return gold_softmax(q_values)

