#!/usr/bin/env python3
"""Infer one TaintInduce rule, then score it against a held-out ground truth.

TaintInduce infers, for every input-bit-to-output-bit flow, a boolean DNF over
the input state.  Its own validator (``validate_rule_explains_observations``)
checks that the inferred rule reproduces the observations it was *trained* on,
which a rule that memorises its training set passes trivially.  RQ1 asks the
other question: does the rule still hold on states it has never seen?

So we infer on one set of seed states and score on a second, freshly drawn one.
The scoring is the tool's own semantics, taken from its validator:

  * the ground truth for input bit ``i`` at state ``s`` is the set of output bits
    that change when bit ``i`` of ``s`` is flipped and the instruction is
    re-executed -- a one-bit noninterference oracle on the real CPU;
  * the rule's prediction is the union of ``pair.output_bit`` over every
    condition-dataflow pair whose ``input_bit`` is ``i`` and whose condition the
    mutated input state satisfies.

We differ from the tool's validator in one deliberate way.  It scores a flow as
explained only on set *equality*, which conflates the two failure directions.
Soundness is a containment property, so we separate them:

  * ground truth minus prediction is non-empty  ->  UNDER-taint.  Unsound: a real
    dependency the engine would miss.  This is the failure RQ1 is about.
  * prediction minus ground truth is non-empty  ->  over-taint.  Imprecise but
    safe.

A flow is *exact* when both differences are empty.  Over-taint is reported split
by what it lands on, because the two kinds mean different things: spurious flows
onto *flag* bits are the ZF/PF ceiling of the paper's Case 3 (a DNF over the
input bits cannot say "only when the result is zero" without becoming
exponential, so the inference emits an unconditional flow), while spurious flows
onto *data* bits are ordinary imprecision.  Table 3's "correct" is the first
case: no dependency missed, over-approximation confined to flags.

Usage:
    score_rule.py --arch X86 --bytes 01d8 --name 'add eax, ebx'

    # re-score a rule a previous run already inferred, which is free
    score_rule.py --arch X86 --bytes 01d8 --rule results/<stamp>/rules/01d8_X86_rule.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# The taintinduce package only exists inside the checkout that
# setup_taintinduce.sh creates, and this script is meant to be run by that
# checkout's interpreter (see run_rq1.sh).  Fail with an instruction, not a
# traceback, when it is not.
try:
    from taintinduce.cpu.cpu import CPUFactory
    from taintinduce.inference_engine.inference import infer
    from taintinduce.inference_engine.observation_processor import (
        extract_observation_dependencies,
    )
    from taintinduce.inference_engine.validation import check_condition_satisfied
    from taintinduce.observation_engine.observation import decode_instruction_bytes
    from taintinduce.state.state import Observation
    from taintinduce.state.state_utils import regs2bits
    from taintinduce.taintinduce import gen_insninfo, gen_obs
except ImportError:  # pragma: no cover
    sys.exit(
        'cannot import taintinduce.\n'
        'Run ./setup_taintinduce.sh first, then use run_rq1.sh, which invokes\n'
        'this script with the checkout\'s own interpreter.',
    )

FLAG_REGISTERS = frozenset({'EFLAGS', 'RFLAGS', 'NZCV', 'CPSR'})

# Which bit of EFLAGS is which, so an under-tainted flag can be named rather than
# reported as "bit 11 of EFLAGS".  Intel SDM Vol.1 3.4.3.
X86_FLAG_BITS = {0: 'CF', 2: 'PF', 4: 'AF', 6: 'ZF', 7: 'SF', 11: 'OF'}
# JN packs its flags NZCV from bit 3 down to bit 0.
JN_FLAG_BITS = {0: 'V', 1: 'C', 2: 'Z', 3: 'N'}


def bit_layout(state_format):
    """Global bit index -> (register name, bit within register).

    TaintInduce numbers the whole machine state as one bit vector, laid out by
    concatenating the registers of ``state_format`` in order (``regs2bits``).
    Rules are expressed in those global indices, so every report has to be
    translated back through this map to be readable.
    """
    layout = {}
    offset = 0
    for reg in state_format:
        for bit in range(reg.bits):
            layout[offset + bit] = (reg.name, bit)
        offset += reg.bits
    return layout


def name_bit(pos, layout, arch):
    """Render a global bit index as e.g. 'EAX[3]' or 'EFLAGS[0]=CF'."""
    reg, bit = layout.get(pos, ('?', pos))
    flags = JN_FLAG_BITS if arch == 'JN' else X86_FLAG_BITS
    if reg in FLAG_REGISTERS and bit in flags:
        return f'{reg}[{bit}]={flags[bit]}'
    return f'{reg}[{bit}]'


def gen_holdout_observations(arch, bytestring, state_format, n_states, seed):
    """Draw fresh seed states and take a one-bit-flip observation at each.

    This mirrors the engine's own ``_gen_observation_worker`` exactly -- flip one
    input bit, re-execute, keep the (before, after) pair -- but runs in this
    process with a seeded RNG, so the held-out set is reproducible and provably
    independent of the seeds the inference was given.
    """
    random.seed(seed)
    cpu = CPUFactory.create_cpu(arch)
    cpu.set_memregs({r for r in state_format if 'MEM' in r.name})
    bytecode = decode_instruction_bytes(bytestring, arch)

    observations = []
    for _ in range(n_states):
        # Draw a state the instruction actually runs on.  Some random states
        # fault (a division by zero, an unmapped pointer); redraw rather than
        # score against an exception.
        for _attempt in range(255):
            try:
                cpu.randomize_regs()
                seed_before, seed_after = cpu.execute(bytecode)
                break
            except Exception:  # noqa: BLE001 - UcError, OutOfRangeException, ...
                continue
        else:
            continue

        seed_in = dict(seed_before)
        seed_state = regs2bits(seed_before, state_format)
        result_state = regs2bits(seed_after, state_format)

        mutated = []
        flipped_positions = set()
        for reg in state_format:
            if 'WRITE' in reg.name or 'ADDR' in reg.name:
                continue
            for bit in range(reg.bits):
                cpu.set_cpu_state(seed_in)
                cpu.write_reg(reg, seed_in[reg] ^ (1 << bit))
                try:
                    before, after = cpu.execute(bytecode)
                except Exception:  # noqa: BLE001
                    continue
                state_before = regs2bits(before, state_format)
                # Keep only mutations that really are a single-bit perturbation
                # of the seed.  A write can land differently from what was asked
                # -- a reserved EFLAGS bit that the CPU normalises away, or one
                # it forces on -- and the dependency extractor rejects an
                # observation whose input differs from the seed in anything other
                # than exactly one, not-yet-seen bit.  Dropping those here costs
                # a few flows and keeps the case from dying on an exception.
                delta = seed_state.diff(state_before)
                if len(delta) != 1:
                    continue
                position = next(iter(delta))
                if position in flipped_positions:
                    continue
                flipped_positions.add(position)
                mutated.append((state_before, regs2bits(after, state_format)))

        observations.append(
            Observation(
                (seed_state, result_state),
                frozenset(mutated),
                bytestring,
                arch,
                state_format,
            ),
        )
    return observations


def register_offsets(state_format):
    """Register name -> (global bit offset, width)."""
    offsets = {}
    position = 0
    for reg in state_format:
        offsets[reg.name] = (position, reg.bits)
        position += reg.bits
    return offsets


def infer_carry_operands(rule, state_format):
    """Guess the destination and source registers from the rule's own pairs.

    The destination of a two-operand ALU instruction is the data register that
    receives a flow from a *different* register.  For `add al, bl` the rule has
    EBX[i] -> EAX[j] pairs and no EAX[i] -> EBX[j] pairs, so EAX is the
    destination and both EAX and EBX are sources.
    """
    layout = bit_layout(state_format)
    data = [r.name for r in state_format if r.name not in FLAG_REGISTERS and 'MEM' not in r.name]
    destinations = set()
    for pair in rule.pairs:
        src_reg = layout.get(pair.input_bit, ('?', 0))[0]
        dst_reg = layout.get(pair.output_bit, ('?', 0))[0]
        if src_reg != dst_reg and src_reg in data and dst_reg in data:
            destinations.add(dst_reg)
    return sorted(destinations), data


def carry_structure(rule, state_format, width, shape, dst_names, src_names):
    """Check the rule's dataflow *shape*, without sampling any state at all.

    This is the cheap, decisive form of the soundness question, and it needs no
    ground truth beyond the algebra of the instruction:

      * ``triangle`` (add, sub).  A carry (or a borrow) out of bit ``i`` travels
        upward, so bit ``i`` of either operand can reach output bit ``j`` for
        *every* ``j >= i``.  The rule therefore needs one dataflow per ``(i, j)``
        with ``j >= i``: ``w(w+1)/2`` of them per source operand.  A missing pair
        is an under-taint that no choice of runtime state can repair, because the
        rule simply has no clause that could fire.  A pair with ``j < i`` is a
        flow to a lower bit, which addition cannot produce.
      * ``diagonal`` (and, or, xor).  Bit ``i`` reaches bit ``i`` and nothing
        else, so the required set is the diagonal and anything off it is
        structural over-taint.

    Note this is a *lower bound* on the damage: a pair can be present and still
    never fire, because its DNF condition is never satisfied.  We count it as
    present anyway, so every hole reported here is real.
    """
    offsets = register_offsets(state_format)
    present = {(pair.input_bit, pair.output_bit) for pair in rule.pairs}

    report = {'shape': shape, 'width': width, 'dst': dst_names, 'src': src_names, 'per_source': []}
    total_required = total_present = total_extra = 0
    # Coverage by carry distance d = j - i, which is the number this experiment
    # exists to show: short carries are learned, long ones are not.
    by_distance = {d: [0, 0] for d in range(width)}

    for dst in dst_names:
        if dst not in offsets:
            continue
        dst_base, dst_bits = offsets[dst]
        for src in src_names:
            if src not in offsets:
                continue
            src_base, src_bits = offsets[src]
            span = min(width, dst_bits, src_bits)
            required = missing = 0
            holes = []
            for i in range(span):
                lo, hi = (i, span) if shape == 'triangle' else (i, i + 1)
                for j in range(lo, hi):
                    required += 1
                    by_distance[j - i][1] += 1
                    if (src_base + i, dst_base + j) in present:
                        by_distance[j - i][0] += 1
                    else:
                        missing += 1
                        holes.append((i, j))
            # Flows the algebra forbids: downward for a triangle, off-diagonal
            # for a bitwise op.
            extra = 0
            for i in range(span):
                for j in range(span):
                    forbidden = (j < i) if shape == 'triangle' else (j != i)
                    if forbidden and (src_base + i, dst_base + j) in present:
                        extra += 1
            total_required += required
            total_present += required - missing
            total_extra += extra
            report['per_source'].append(
                {
                    'src': src,
                    'dst': dst,
                    'required': required,
                    'present': required - missing,
                    'missing': missing,
                    'forbidden_present': extra,
                    'holes': [f'{i}->{j}' for i, j in holes[:24]],
                    'span': span,
                },
            )

    report['required'] = total_required
    report['present'] = total_present
    report['missing'] = total_required - total_present
    report['forbidden_present'] = total_extra
    report['by_distance'] = {d: by_distance[d] for d in range(width) if by_distance[d][1]}
    return report


def print_carry_structure(report):
    shape, width = report['shape'], report['width']
    label = 'triangle (carries travel up)' if shape == 'triangle' else 'diagonal (bit i to bit i only)'
    print(f'carry structure, width {width}, expected shape: {label}', flush=True)
    print(
        f'  required dataflows {report["required"]}, present {report["present"]}, '
        f'MISSING {report["missing"]}, forbidden-but-present {report["forbidden_present"]}',
        flush=True,
    )
    for row in report['per_source']:
        print(
            f'  {row["src"]}[i] -> {row["dst"]}[j]: {row["present"]}/{row["required"]}'
            f'   missing {row["missing"]}   forbidden-present {row["forbidden_present"]}',
            flush=True,
        )
        if row['holes']:
            print(f'    first holes (i->j): {" ".join(row["holes"])}', flush=True)
    if report['by_distance']:
        cells = [
            f'd={d} {have}/{need}'
            for d, (have, need) in sorted(report['by_distance'].items())
            if need
        ]
        print('  by carry distance: ' + '  '.join(cells), flush=True)


def print_carry_matrix(report, present_pairs, offsets):
    """Draw the (i, j) grid for narrow operands, where it fits on a line."""
    width = report['width']
    if width > 16:
        return
    for row in report['per_source']:
        src_base = offsets[row['src']][0]
        dst_base = offsets[row['dst']][0]
        span = row['span']
        header = ''.join(f'{j % 10}' for j in range(span))
        print(f'  {row["src"]}[i] -> {row["dst"]}[j]      j: {header}', flush=True)
        for i in range(span):
            line = []
            for j in range(span):
                required = (j >= i) if report['shape'] == 'triangle' else (j == i)
                here = (src_base + i, dst_base + j) in present_pairs
                if required:
                    line.append('#' if here else '.')
                else:
                    line.append('x' if here else ' ')
            print(f'    {"i=%-2d" % i}            {"".join(line)}', flush=True)
    print('    legend: # required and present, . REQUIRED AND MISSING, x present but impossible', flush=True)


def carry_witnesses(i, j, width):
    """Candidate operand pairs that force a carry from bit i to bit j.

    For an addition the canonical one is the first: put a single 1 at bit i in
    one operand and a run of 1s across bits i..j-1 in the other.  Bit i then
    generates a carry, every intermediate bit propagates it (1 + 0 + carry = 0
    carry 1), and it lands on bit j.  Clearing bit i clears bit j, so the
    dependency is forced rather than hoped for.

    The rest cover subtraction (a borrow out of a lone set bit runs all the way
    up) and the adc/sbb forms.  We do not decide in advance which one an
    instruction needs: each candidate is executed on the real CPU and kept only
    if the dependency actually shows up (see carry_witness_check).
    """
    full = (1 << width) - 1
    run = ((1 << j) - 1) & ~((1 << i) - 1)  # bits i .. j-1
    one = 1 << i
    return [
        (one, run),
        (run, one),
        (one, one),
        (one, 0),
        (0, one),
        (full, one),
        (one, full),
        (((1 << j) - 1), one),
        (one, ((1 << j) - 1)),
    ]


def carry_witness_check(  # noqa: C901 - one flat sweep over (source, i)
    rule,
    arch,
    bytestring,
    state_format,
    width,
    shape,
    dst_name,
    src_names,
    rng_seed=7,
    random_states=64,
):
    """For every required (i -> j) dataflow, ask whether the rule fires on states that force it.

    Two kinds of witness, and the difference between them is the result:

      * **constructed.**  ``carry_witnesses`` builds the state by hand: a single 1
        at bit i in one operand, a run of 1s across i..j-1 in the other.  Every
        (i, j) gets one, including the long carries.  These states are also
        close in shape to the tool's own Bitwalk and BitFill seed strategies, so
        the inference has very likely trained on something like them, and they
        should be read as the friendly case.
      * **random.**  Uniformly drawn operand values, one bit flipped.  One run
        exposes every j that bit i reaches at once, which is what makes this
        cheap.  It is also where the rule breaks: the condition attached to a
        dataflow is a DNF fitted to the sampled states, and an unseen state
        walks straight out of it.

    Random draws cannot witness the long carries at all: a carry from bit 0 to
    bit 31 needs all thirty intermediate bit-pairs configured to propagate, so
    it turns up with probability about 2^-31.  Cells with no random witness are
    reported as such rather than counted either way, and that emptiness in the
    far corner of the triangle is the paper's argument in visual form.
    """
    rng = random.Random(rng_seed)
    cpu = CPUFactory.create_cpu(arch)
    cpu.set_memregs({r for r in state_format if 'MEM' in r.name})
    code = decode_instruction_bytes(bytestring, arch)
    offsets = register_offsets(state_format)
    dst_base, dst_bits = offsets[dst_name]

    def run_pair(values, src_name, bit):
        """Execute with these operands and with bit `bit` of `src_name` flipped.

        Returns (reached, mutated_state, base_state) where `reached` is the set of
        output bit indices *within the destination register* that differ, or None
        if the pair is not a clean one-bit perturbation.
        """
        base = {reg: values.get(reg.name, 0) for reg in state_format}
        flipped = dict(base)
        for reg in state_format:
            if reg.name == src_name:
                flipped[reg] = values.get(src_name, 0) ^ (1 << bit)
        try:
            cpu.set_cpu_state(base)
            before_a, after_a = cpu.execute(code)
            cpu.set_cpu_state(flipped)
            before_b, after_b = cpu.execute(code)
        except Exception:  # noqa: BLE001 - a faulting state is simply unusable
            return None, None, None
        state_a, state_b = regs2bits(before_a, state_format), regs2bits(before_b, state_format)
        delta_in = state_a.diff(state_b)
        if len(delta_in) != 1 or next(iter(delta_in)) != offsets[src_name][0] + bit:
            return None, None, None
        out = regs2bits(after_a, state_format).diff(regs2bits(after_b, state_format))
        reached = {b - dst_base for b in out if dst_base <= b < dst_base + dst_bits}
        return reached, state_b, state_a

    def rule_reaches(src_name, bit, state_b, state_a):
        """The output bits of the destination the rule taints from this input bit."""
        base = offsets[src_name][0] + bit
        got = predict(rule.pairs, base, state_b) | predict(rule.pairs, base, state_a)
        return {b - dst_base for b in got if dst_base <= b < dst_base + dst_bits}

    report = {
        'width': width,
        'shape': shape,
        'dst': dst_name,
        'random_states': random_states,
        'per_source': [],
    }

    for src_name in src_names:
        if src_name not in offsets:
            continue
        src_bits = offsets[src_name][1]
        span = min(width, src_bits, dst_bits)
        others = [n for n in src_names if n != src_name] or [src_name]

        cells = {}  # 'i,j' -> dict.  String keys, because this goes to JSON.
        for i in range(span):
            targets = list(range(i, span)) if shape == 'triangle' else [i]
            for j in targets:
                cells[f'{i},{j}'] = {'forced': 'none', 'rand_seen': 0, 'rand_fires': 0}

            # --- constructed witnesses, one per (i, j) ---------------------
            for j in targets:
                for src_value, other_value in carry_witnesses(i, j, span):
                    values = {src_name: src_value}
                    for other in others:
                        if other != src_name:
                            values[other] = other_value
                    reached, state_b, state_a = run_pair(values, src_name, i)
                    if reached is None or j not in reached:
                        continue
                    fired = j in rule_reaches(src_name, i, state_b, state_a)
                    cells[f'{i},{j}']['forced'] = 'fires' if fired else 'silent'
                    if fired:
                        break

            # --- random witnesses, all j at once --------------------------
            for _ in range(random_states):
                values = {src_name: rng.getrandbits(span)}
                for other in others:
                    if other != src_name:
                        values[other] = rng.getrandbits(span)
                reached, state_b, state_a = run_pair(values, src_name, i)
                if reached is None:
                    continue
                fired = rule_reaches(src_name, i, state_b, state_a)
                for j in targets:
                    if j in reached:
                        cells[f'{i},{j}']['rand_seen'] += 1
                        if j in fired:
                            cells[f'{i},{j}']['rand_fires'] += 1

        report['per_source'].append({'src': src_name, 'span': span, 'cells': cells})

    # Roll the cells up.  Every count is stated against what was actually
    # witnessed; a cell nothing could witness is counted as witnessed by nothing
    # and claimed for neither side.
    totals = {
        'required': 0,
        'forced_fires': 0,
        'forced_silent': 0,
        'forced_unwitnessed': 0,
        'rand_cells_seen': 0,
        'rand_cells_always': 0,
        'rand_cells_partial': 0,
        'rand_cells_never': 0,
        'rand_cells_unseen': 0,
        'rand_witnesses': 0,
        'rand_witness_fires': 0,
    }
    for row in report['per_source']:
        for cell in row['cells'].values():
            totals['required'] += 1
            totals['forced_' + {'fires': 'fires', 'silent': 'silent', 'none': 'unwitnessed'}[cell['forced']]] += 1
            totals['rand_witnesses'] += cell['rand_seen']
            totals['rand_witness_fires'] += cell['rand_fires']
            if cell['rand_seen'] == 0:
                totals['rand_cells_unseen'] += 1
            else:
                totals['rand_cells_seen'] += 1
                if cell['rand_fires'] == cell['rand_seen']:
                    totals['rand_cells_always'] += 1
                elif cell['rand_fires'] == 0:
                    totals['rand_cells_never'] += 1
                else:
                    totals['rand_cells_partial'] += 1
    report['totals'] = totals
    return report


def cell_symbol(cell):
    """One character per (i, j) dataflow, for the triangle picture."""
    if cell['rand_seen'] == 0:
        # No random state exercised this dataflow.  Fall back to what the
        # constructed witness said.
        return {'fires': 'o', 'silent': 'X', 'none': '?'}[cell['forced']]
    if cell['rand_fires'] == cell['rand_seen']:
        return '#'
    if cell['rand_fires'] == 0:
        return '.'
    return '~'


def print_carry_witness(report):
    t = report['totals']
    print(
        f'forced-carry check, width {report["width"]}, destination {report["dst"]}, '
        f'{report["random_states"]} random states per input bit',
        flush=True,
    )
    print(
        f'  {t["required"]} required dataflows.  On CONSTRUCTED carry chains: '
        f'{t["forced_fires"]} fire, {t["forced_silent"]} silent, {t["forced_unwitnessed"]} unwitnessed.',
        flush=True,
    )
    print(
        f'  On RANDOM states: {t["rand_cells_seen"]} dataflows were witnessed at all '
        f'({t["rand_cells_unseen"]} never turned up in a random draw) -- of those, '
        f'{t["rand_cells_always"]} always fire, {t["rand_cells_partial"]} fire only sometimes, '
        f'{t["rand_cells_never"]} never fire.',
        flush=True,
    )
    if t['rand_witnesses']:
        pct = 100.0 * t['rand_witness_fires'] / t['rand_witnesses']
        print(
            f'  over every random witness: {t["rand_witness_fires"]}/{t["rand_witnesses"]} '
            f'({pct:.1f}%) of forced dataflows are tainted by the rule',
            flush=True,
        )
    for row in report['per_source']:
        span = row['span']
        if span > 32:
            continue
        print(f'  {row["src"]}[i] -> {report["dst"]}[j]', flush=True)
        print('      j: ' + ''.join(f'{j % 10}' for j in range(span)), flush=True)
        for i in range(span):
            line = ''.join(
                cell_symbol(row['cells'][f'{i},{j}']) if f'{i},{j}' in row['cells'] else ' '
                for j in range(span)
            )
            print(f'    i={i:<2}  {line}', flush=True)
    print(
        '    legend: # always fires   ~ sometimes   . never, though witnessed   '
        'o only a constructed witness, and it fires   X constructed witness, silent   ? unwitnessed',
        flush=True,
    )


def predict(rule_pairs, input_bit, input_state):
    """The rule's output-taint prediction for one tainted input bit."""
    return {
        pair.output_bit
        for pair in rule_pairs
        if pair.input_bit == input_bit and check_condition_satisfied(pair.condition, input_state)
    }


def score(rule, observations, state_format, arch, max_examples=8):
    """Compare the rule against the held-out one-bit dependencies."""
    layout = bit_layout(state_format)
    flag_bits = {pos for pos, (reg, _) in layout.items() if reg in FLAG_REGISTERS}
    # Which register each output bit belongs to, so a missed carry can be
    # recognised: same register as the input, higher bit index.
    reg_of = {pos: reg for pos, (reg, _) in layout.items()}
    idx_of = {pos: bit for pos, (_, bit) in layout.items()}

    counts = {
        'flows': 0,
        'exact': 0,
        'under': 0,
        'over': 0,
        'under_bits': 0,
        'over_bits': 0,
        'flows_flags': 0,
        'under_flags': 0,
        'flows_data': 0,
        'under_data': 0,
        'over_data': 0,
        'over_flags': 0,
        'under_carry_like': 0,
    }
    examples = []
    over_examples = []

    for obs_dep in extract_observation_dependencies(observations):
        for input_bit, truth in obs_dep.dataflow.items():
            input_state = obs_dep.mutated_inputs.get_input_state(input_bit)
            predicted = predict(rule.pairs, input_bit, input_state)
            missing = set(truth) - predicted
            extra = predicted - set(truth)

            counts['flows'] += 1
            # A flow is counted as touching flags if the truth includes any flag
            # bit; the same flow can also touch data bits.
            truth_flags = {b for b in truth if b in flag_bits}
            truth_data = set(truth) - truth_flags
            if truth_flags:
                counts['flows_flags'] += 1
                if missing & flag_bits:
                    counts['under_flags'] += 1
            if truth_data:
                counts['flows_data'] += 1
                if missing - flag_bits:
                    counts['under_data'] += 1

            if missing:
                counts['under'] += 1
                counts['under_bits'] += len(missing)
                # A missed carry: the dependency runs from a bit to a *higher*
                # bit of the same register, which is what carry propagation is
                # and what a finite seed set is exponentially unlikely to cover.
                if any(
                    reg_of.get(m) == reg_of.get(input_bit) and idx_of.get(m, -1) > idx_of.get(input_bit, 1 << 30)
                    for m in missing
                ):
                    counts['under_carry_like'] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            'input': name_bit(input_bit, layout, arch),
                            'state': f'0x{input_state.state_value:x}',
                            'missing': sorted(name_bit(m, layout, arch) for m in missing),
                            'predicted': sorted(name_bit(p, layout, arch) for p in predicted),
                        },
                    )
            if extra:
                counts['over'] += 1
                counts['over_bits'] += len(extra)
                # Split the over-taint by what it lands on.  Spurious *flag* bits
                # are the expected shape here: ZF and PF are functions of the
                # whole result, and a DNF over the input bits cannot say
                # "the result happens to be zero" without becoming exponential,
                # so the inference falls back to an unconditional flow.
                if extra - flag_bits:
                    counts['over_data'] += 1
                else:
                    counts['over_flags'] += 1
                if len(over_examples) < max_examples:
                    over_examples.append(
                        {
                            'input': name_bit(input_bit, layout, arch),
                            'state': f'0x{input_state.state_value:x}',
                            'spurious': sorted(name_bit(e, layout, arch) for e in extra),
                        },
                    )
            if not missing and not extra:
                counts['exact'] += 1

    return counts, examples, over_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arch', required=True, choices=['X86', 'AMD64', 'ARM64', 'JN'])
    parser.add_argument('--bytes', required=True, help='instruction encoding in hex, e.g. 01d8')
    parser.add_argument('--name', default='', help='human label for the report, e.g. "add eax, ebx"')
    parser.add_argument('--infer-seed', type=int, default=1, help='RNG seed for the training observations')
    parser.add_argument('--holdout-seed', type=int, default=1000, help='RNG seed for the held-out observations')
    parser.add_argument('--holdout-states', type=int, default=32, help='number of fresh seed states to score on')
    parser.add_argument('--obs-dir', default='', help='directory to keep the observations and rule in')
    parser.add_argument(
        '--rule',
        default='',
        help='score a rule that a previous run already wrote, instead of inferring a new one. '
        'Inference is the expensive half, so this is how you re-examine a result without paying for it twice.',
    )
    parser.add_argument('--json', default='', help='write the verdict here as JSON')
    parser.add_argument(
        '--carry-width',
        type=int,
        default=0,
        help='operand width in bits; enables the structural dataflow-shape check. '
        '8 for `add al, bl`, 32 for `add eax, ebx`, 4 for JN.',
    )
    parser.add_argument(
        '--carry-shape',
        default='triangle',
        choices=['triangle', 'diagonal'],
        help='triangle for additive instructions (bit i reaches every j >= i), '
        'diagonal for bitwise ones (bit i reaches bit i and nothing else)',
    )
    parser.add_argument('--carry-dst', default='', help='destination register; inferred from the rule when omitted')
    parser.add_argument('--carry-src', default='', help='comma-separated source registers; inferred when omitted')
    args = parser.parse_args()

    label = args.name or f'{args.bytes} ({args.arch})'
    print(f'=== {label} [{args.arch} {args.bytes}] ===', flush=True)

    insn = gen_insninfo(args.arch, args.bytes)
    state_format = insn.state_format
    state_bits = sum(reg.bits for reg in state_format)
    layout_desc = ' '.join(f'{r.name}:{r.bits}' for r in state_format)
    # Below 2^14 states the observation engine enumerates the state space instead
    # of sampling it, which is the single most important fact about this
    # experiment: it is the line between "the inference has seen everything" and
    # "the inference is generalising from samples".
    exhaustive = state_bits < 14
    print(f'state: {state_bits} bits ({layout_desc}) -> seeds are {"EXHAUSTIVE" if exhaustive else "SAMPLED"}', flush=True)

    if args.rule:
        from taintinduce.serialization import TaintInduceDecoder  # noqa: PLC0415

        with open(args.rule) as fh:
            rule = json.load(fh, cls=TaintInduceDecoder)
        n_observations, t_observe, t_infer = 0, 0.0, 0.0
        print(f're-scoring the rule in {args.rule} ({len(rule.pairs)} rule pairs)', flush=True)
    else:
        random.seed(args.infer_seed)
        t0 = time.time()
        observations, _engine = gen_obs(args.arch, args.bytes, state_format)
        t_observe = time.time() - t0

        t0 = time.time()
        rule = infer(observations, output_induction=False)
        t_infer = time.time() - t0
        n_observations = len(observations)
        print(
            f'inferred in {t_observe:.1f}s observing + {t_infer:.1f}s minimising '
            f'({n_observations} observations, {len(rule.pairs)} rule pairs)',
            flush=True,
        )

        if args.obs_dir:
            os.makedirs(args.obs_dir, exist_ok=True)
            from taintinduce.serialization import TaintInduceEncoder  # noqa: PLC0415

            with open(os.path.join(args.obs_dir, f'{args.bytes}_{args.arch}_rule.json'), 'w') as fh:
                json.dump(rule.convert2squirrel(args.arch, args.bytes), fh, cls=TaintInduceEncoder, indent=2)

    carry = None
    witness = None
    if args.carry_width:
        inferred_dst, data_regs = infer_carry_operands(rule, state_format)
        dst_names = [d for d in args.carry_dst.split(',') if d] or inferred_dst
        src_names = [x for x in args.carry_src.split(',') if x] or data_regs
        if not dst_names:
            print('!! could not identify a destination register: skipping the structural check', flush=True)
        else:
            carry = carry_structure(rule, state_format, args.carry_width, args.carry_shape, dst_names, src_names)
            print_carry_structure(carry)
            print_carry_matrix(
                carry,
                {(pair.input_bit, pair.output_bit) for pair in rule.pairs},
                register_offsets(state_format),
            )
            # The pair being present is necessary, not sufficient: each one
            # carries a DNF condition that decides whether it fires.  Force each
            # dependency on the CPU and see.
            t_witness = time.time()
            witness = carry_witness_check(
                rule,
                args.arch,
                args.bytes,
                state_format,
                args.carry_width,
                args.carry_shape,
                dst_names[0],
                src_names,
            )
            witness['seconds'] = time.time() - t_witness
            print_carry_witness(witness)

    t0 = time.time()
    holdout = gen_holdout_observations(
        args.arch,
        args.bytes,
        state_format,
        args.holdout_states,
        args.holdout_seed,
    )
    counts, examples, over_examples = score(rule, holdout, state_format, args.arch)
    t_score = time.time() - t0

    if counts['flows'] == 0:
        # An oracle that compares nothing agrees with everything.  Say so loudly
        # instead of printing a green verdict.
        verdict = 'NO-DATA'
        print('!! the held-out oracle produced zero flows: nothing was compared', flush=True)
    elif counts['under'] > 0:
        verdict = 'UNSOUND'
    elif counts['over_data'] > 0:
        verdict = 'sound, over-taints data'
    elif counts['over'] > 0:
        # Only flag bits are over-tainted.  Table 3 calls these families correct:
        # every real dependency is present and the dependency structure on the
        # data path is exact.
        verdict = 'correct (over-taints flags)'
    else:
        verdict = 'correct'

    print(
        f'held-out: {counts["flows"]} flows over {len(holdout)} fresh states, scored in {t_score:.1f}s',
        flush=True,
    )
    print(
        f'  exact {counts["exact"]}  under-taint {counts["under"]} '
        f'({counts["under_bits"]} bits)  over-taint {counts["over"]} ({counts["over_bits"]} bits)',
        flush=True,
    )
    if counts['flows_data']:
        print(
            f'  data-register flows: {counts["flows_data"]}, under-tainted {counts["under_data"]}'
            f'   (of which carry-shaped: {counts["under_carry_like"]})',
            flush=True,
        )
    if counts['flows_flags']:
        print(f'  flag flows: {counts["flows_flags"]}, under-tainted {counts["under_flags"]}', flush=True)
    print(f'  over-taint lands on: {counts["over_data"]} flows touching data, {counts["over_flags"]} flags only', flush=True)
    for ex in examples:
        print(f'    missed  {ex["input"]} -> {", ".join(ex["missing"])}  at state {ex["state"]}', flush=True)
    for ex in over_examples:
        print(f'    spurious {ex["input"]} -> {", ".join(ex["spurious"])}  at state {ex["state"]}', flush=True)
    if witness:
        t = witness['totals']
        broken = t['rand_cells_partial'] + t['rand_cells_never']
        if broken:
            print(
                f'FORCED-CARRY {label}: of the {t["rand_cells_seen"]} required dataflows a random '
                f'state ever exercises, {broken} are not always tainted '
                f'({t["rand_cells_never"]} never are); {t["rand_cells_unseen"]} of '
                f'{t["required"]} were never exercised at all',
                flush=True,
            )
        else:
            print(
                f'FORCED-CARRY {label}: every one of the {t["rand_cells_seen"]} witnessed '
                f'dataflows is always tainted',
                flush=True,
            )
    if carry and carry['missing']:
        # This one does not depend on which states we happened to draw: the rule
        # has no clause at all for those flows, so no runtime state can make it
        # taint them.
        print(
            f'STRUCTURE {label}: {carry["missing"]} of {carry["required"]} required dataflows '
            f'are absent from the rule',
            flush=True,
        )
    elif carry:
        print(f'STRUCTURE {label}: all {carry["required"]} required dataflows present', flush=True)
    print(f'VERDICT {label}: {verdict}', flush=True)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(
                {
                    'name': label,
                    'arch': args.arch,
                    'bytes': args.bytes,
                    'state_bits': state_bits,
                    'seeds_exhaustive': exhaustive,
                    'observations': n_observations,
                    'rule_pairs': len(rule.pairs),
                    'seconds': {'observe': t_observe, 'infer': t_infer, 'score': t_score},
                    'holdout_states': len(holdout),
                    'holdout_seed': args.holdout_seed,
                    'infer_seed': args.infer_seed,
                    'counts': counts,
                    'carry': carry,
                    'forced_carry': witness,
                    'examples': examples,
                    'over_examples': over_examples,
                    'verdict': verdict,
                },
                fh,
                indent=2,
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
