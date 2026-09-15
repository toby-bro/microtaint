# ruff: noqa: W505, E501
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
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
    except Exception:
        return None


def unicorn_trace(gt, code, state, flags):
    """[(addr, size)] Unicorn actually executed, or None if it did not complete."""
    uc = gt._fresh(code)
    seen = []
    h = uc.hook_add(UC_HOOK_CODE, lambda _u, a, s, _d: seen.append((a, s)))
    try:
        gt._run(uc, code, state, flags)
    except Exception:
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
    except Exception:
        return {'checked': False}
    # Re-run without restoring, so whatever the last run left behind is live.
    try:
        dirty = super(type(gt), gt)._run(uc, code, state, flags)
    except Exception:
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
