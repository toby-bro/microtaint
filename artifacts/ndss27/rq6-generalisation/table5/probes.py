# ruff: noqa: W505, E501, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_table5_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Probes that decide whether a finding can be trusted, without hardcoded lists.

`model_closure` is the general form of the `shrd rbp, rsp` bug.  The oracle
models four GPRs, SP and the flags; any instruction that reads or writes anything
else is outside that model, and comparing its taint against the oracle is
meaningless.  Rather than enumerate every register per ISA, the probe runs the
same tracked inputs from two different histories: a clean context, and a context
dirtied by having just executed the same code.  If the outputs differ, the
instruction carries state the oracle does not control, and the entry is
quarantined instead of producing under-taint reports.

`decode_agreement` catches Unicorn executing something other than the instruction
the bytes encode (the known VEX case, where it runs the legacy 2-operand form).
Capstone gives the intended decode; a UC_HOOK_CODE records what Unicorn actually
stepped through.  A disagreement in instruction count or length means Unicorn is
the wrong oracle for those bytes.
"""
from __future__ import annotations

import capstone
from unicorn import UC_HOOK_CODE

CS_ARCH = {
    'x86_64': (capstone.CS_ARCH_X86, capstone.CS_MODE_64),
    'arm64': (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM),
    'mips': (capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS64 | capstone.CS_MODE_BIG_ENDIAN),
    'ppc': (capstone.CS_ARCH_PPC, capstone.CS_MODE_32 | capstone.CS_MODE_BIG_ENDIAN),
    'riscv': (capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV64),
}


def capstone_decode(isa_key, code, addr=0x1000):
    """[(addr, size, text)] as the bytes are *meant* to decode, or None."""
    try:
        arch, mode = CS_ARCH[isa_key]
        md = capstone.Cs(arch, mode)
        out = [(i.address, i.size, f'{i.mnemonic} {i.op_str}'.strip())
               for i in md.disasm(code, addr)]
        return out or None
    except Exception:  # noqa: BLE001 -- capstone gap is not a campaign failure
        return None


def unicorn_trace(gt, code, state, flags):
    """[(addr, size)] Unicorn actually executed, or None if it did not complete."""
    uc = gt._fresh(code)
    seen = []
    h = uc.hook_add(UC_HOOK_CODE, lambda _u, a, s, _d: seen.append((a, s)))
    try:
        gt._run(uc, code, state, flags)
    except Exception:  # noqa: BLE001
        return None
    finally:
        uc.hook_del(h)
    return seen


def decode_agreement(gt, isa_key, code, state, flags):
    """Compare the intended decode against what Unicorn stepped through."""
    cs = capstone_decode(isa_key, code)
    uc = unicorn_trace(gt, code, state, flags)
    if cs is None or uc is None:
        return {'checked': False, 'agree': None, 'capstone': cs, 'unicorn': uc}
    agree = ([s for _, s in uc] == [s for _, s, _ in cs])
    return {'checked': True, 'agree': agree,
            'capstone': [(hex(a), s, t) for a, s, t in cs],
            'unicorn': [(hex(a), s) for a, s in uc]}


def model_closure(gt, code, state, flags):
    """Does this instruction depend on state the oracle does not model?

    clean : pristine context (every untracked register 0)
    again : pristine context a second time  -> must equal `clean` (determinism)
    dirty : the context left behind by a previous execution of the same code
            -> differs from `clean` only if untracked state feeds the result
    """
    uc = gt._fresh(code)
    try:
        clean = gt._run(uc, code, state, flags)
        again = gt._run(uc, code, state, flags)
    except Exception:  # noqa: BLE001
        return {'checked': False}
    # Re-run without restoring, so whatever the last run left behind is live.
    try:
        dirty = super(type(gt), gt)._run(uc, code, state, flags)
    except Exception:  # noqa: BLE001
        return {'checked': True, 'deterministic': clean == again,
                'closed': False, 'note': 'dirty run faulted'}
    return {
        'checked': True,
        'deterministic': clean == again,
        'closed': dirty == clean,
        'leaked': sorted(r for r in clean if clean[r] != dirty.get(r)),
    }


def model_closure_sampled(gt, code, states, flags):
    """`model_closure` over many states.

    One state is not enough.  The probe can only compare registers the oracle
    returns, so leaked state stays invisible until it happens to perturb a
    tracked output or a flag: `shrd rbp, rsp` leaks RBP for every state, yet
    looks closed for the ones where the leaked bits shift out of the flag
    computation.  An entry is open if ANY sampled state shows the leak.
    """
    det = True
    leaked_any = False
    leaked_regs: set[str] = set()
    checked = 0
    for st in states:
        r = model_closure(gt, code, st, flags)
        if not r.get('checked'):
            continue
        checked += 1
        det &= bool(r.get('deterministic', True))
        if not r.get('closed', True):
            leaked_any = True
            leaked_regs.update(r.get('leaked', ()))
    return {'checked': checked, 'deterministic': det,
            'closed': not leaked_any, 'leaked': sorted(leaked_regs)}


#: States for the preserved-flag test.  Far more than PROBE_STATES because it
#: costs ONE run per state (not one per register) and because a sparse sample is
#: exactly how an UNDEFINED flag gets mistaken for a preserved one.
PRESERVED_STATES = 24

def executability(gt, code, cases):
    """How many sampled states this form can actually be executed at.

    An entry Unicorn cannot run is not a passing test, it is no test.  x86
    `adcx`/`adox` raise UC_ERR_INSN_INVALID (no ADX support) and MIPS
    `clz`/`clo`/`dclz`/`dclo`/`movz`/`movn` raise UC_ERR_EXCEPTION, yet the
    latter reported checked > 0 at 100% bit-exact.  Decided by running the
    instruction, so it needs no list and cannot go stale.
    """
    ran = 0
    for st, fl in cases:
        try:
            gt._run(gt._fresh(code), code, st, fl)
            ran += 1
        except Exception:  # noqa: BLE001 -- not executing IS the result here
            pass
    return {'ran': ran, 'of': len(cases)}


def undeclared_sources(gt, isa, code, srcs, cases):
    """Modelled registers that change a scored output but are not in `srcs`.

    `srcs` is hand-written per corpus entry and decides what may be tainted, so
    an omission makes an operand structurally untestable with no error: the
    shift count of `shl rax, cl [cl=1]` and the divisor of PPC `divw 3,4,5` are
    never tainted today.  Rather than trust the declaration, perturb every
    OTHER modelled register and see whether a scored output moves.
    """
    sp = isa.sp[0] if isa.sp else None
    found = set()
    for st, fl in cases:
        try:
            base = gt._run(gt._fresh(code), code, st, fl)
        except Exception:  # noqa: BLE001
            continue
        for name, _ in isa.gprs:
            if name in srcs or name == sp or name in found:
                continue
            st2 = dict(st)
            st2[name] = (st2.get(name, 0) ^ isa.mask) & isa.mask
            try:
                alt = gt._run(gt._fresh(code), code, st2, fl)
            except Exception:  # noqa: BLE001
                continue
            # Ignore the perturbed register itself: it differs by construction.
            if any(alt.get(r) != base.get(r) for r in base if r != name):
                found.add(name)
    return sorted(found)


def preserved_flags(gt, isa, code, cases, srcs):
    """Flags this form READS, and leaves EXACTLY as it found them.

    `written_flags` asks the engine's lift which flags an instruction writes,
    and `chk` scores only those.  That correctly drops flags the ISA leaves
    UNDEFINED, but it also drops flags the ISA PRESERVES, and those carry an
    obligation: `adc x0, x1, x2` reads the carry without writing it, so the
    engine must keep the carry's taint, and nothing checked that it did.  The
    oracle supplies the discriminator the lift cannot: a preserved flag comes
    back bit-identical to its input at every state, an undefined one does not.

    Restricted to DECLARED SOURCES, and that restriction is load-bearing.
    Identity across a sample is not proof of preservation: x86 leaves SF, ZF, AF
    and PF UNDEFINED after `mul`, `imul`, `bsf` and `bsr`, and Unicorn happens to
    leave SF alone often enough that a four-state sample called it preserved.
    Scoring it then produced 259 under-taints in 18,080 cases against an engine
    that is right not to model an undefined flag.  A flag the form actually
    READS is a different matter: its taint has to survive, and that is the
    obligation this check exists to enforce.

    Sampled, so it is evidence rather than proof, which is why it is used only
    to ADD a check on identity behaviour and never to excuse one.
    """
    keep = {f.name for f in isa.flags} & set(srcs)
    if not keep:
        return keep
    for st, fl in cases:
        try:
            out = gt._run(gt._fresh(code), code, st, fl)
        except Exception:  # noqa: BLE001
            continue
        keep &= {n for n in keep if out.get(n) == fl.get(n)}
        if not keep:
            break
    return keep
