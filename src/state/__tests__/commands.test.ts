import { describe, expect, it, vi } from 'vitest';
import {
  CommandStack,
  MAX_UNDO_DEPTH,
  setFlagCommand,
  setParameterCommand,
  structuralCommand,
} from '../commands';

describe('command stack (§6.5)', () => {
  it('applies, undoes and redoes', () => {
    const params = new Float64Array([1, 2, 3]);
    const stack = new CommandStack();
    stack.execute(setParameterCommand('w', params, 1, 2, 9, () => {}));
    expect(params[1]).toBe(9);
    stack.undo();
    expect(params[1]).toBe(2);
    stack.redo();
    expect(params[1]).toBe(9);
  });

  it('reports what undo and redo will do', () => {
    const stack = new CommandStack();
    expect(stack.canUndo).toBe(false);
    expect(stack.undoLabel).toBeNull();

    const params = new Float64Array([0]);
    stack.execute(setParameterCommand('Edit weight', params, 0, 0, 1, () => {}));
    expect(stack.undoLabel).toBe('Edit weight');
    stack.undo();
    expect(stack.canUndo).toBe(false);
    expect(stack.redoLabel).toBe('Edit weight');
  });

  it('clears the redo stack once a new command is executed', () => {
    const params = new Float64Array([0]);
    const stack = new CommandStack();
    stack.execute(setParameterCommand('a', params, 0, 0, 1, () => {}));
    stack.undo();
    expect(stack.canRedo).toBe(true);
    stack.execute(setParameterCommand('b', params, 0, 0, 5, () => {}));
    expect(stack.canRedo).toBe(false);
  });

  it('merges consecutive edits to the same parameter into one undo', () => {
    // Holding an arrow key must not bury the previous edit under a hundred
    // single-step entries.
    const params = new Float64Array([0]);
    const stack = new CommandStack();
    stack.execute(setParameterCommand('nudge', params, 0, 0, 0.1, () => {}));
    stack.execute(setParameterCommand('nudge', params, 0, 0.1, 0.2, () => {}));
    stack.execute(setParameterCommand('nudge', params, 0, 0.2, 0.3, () => {}));
    expect(params[0]).toBeCloseTo(0.3, 12);
    expect(stack.depth).toBe(1);

    // And one undo returns all the way to the original value.
    stack.undo();
    expect(params[0]).toBe(0);
  });

  it('does not merge edits to different parameters', () => {
    const params = new Float64Array([0, 0]);
    const stack = new CommandStack();
    stack.execute(setParameterCommand('a', params, 0, 0, 1, () => {}));
    stack.execute(setParameterCommand('b', params, 1, 0, 1, () => {}));
    expect(stack.depth).toBe(2);
    stack.undo();
    expect(params[1]).toBe(0);
    expect(params[0]).toBe(1);
  });

  it('notifies on every apply and revert, so the canvas repaints', () => {
    const onChange = vi.fn();
    const params = new Float64Array([0]);
    const stack = new CommandStack();
    stack.execute(setParameterCommand('w', params, 0, 0, 1, onChange));
    stack.undo();
    stack.redo();
    expect(onChange).toHaveBeenCalledTimes(3);
  });

  it('caps its depth rather than growing without bound', () => {
    const params = new Float64Array(1);
    const stack = new CommandStack();
    for (let i = 0; i < MAX_UNDO_DEPTH + 50; i++) {
      // Alternating indices so nothing merges.
      stack.execute(setParameterCommand(`e${i}`, params, 0, i, i + 1, () => {}));
      stack.execute(setFlagCommand(`f${i}`, () => false, () => {}, true, () => {}));
    }
    expect(stack.depth).toBeLessThanOrEqual(MAX_UNDO_DEPTH);
  });

  it('undoing an empty stack is a no-op', () => {
    const stack = new CommandStack();
    expect(stack.undo()).toBeNull();
    expect(stack.redo()).toBeNull();
  });

  it('record() logs a command whose effect already happened', () => {
    // Used by the scrub, which mutates live and pushes one entry at mouseup.
    const params = new Float64Array([7]);
    const stack = new CommandStack();
    stack.record(setParameterCommand('scrub', params, 0, 1, 7, () => {}));
    expect(params[0]).toBe(7);
    stack.undo();
    expect(params[0]).toBe(1);
  });
});

describe('flag commands', () => {
  it('round-trips a boolean', () => {
    let frozen = false;
    const stack = new CommandStack();
    stack.execute(
      setFlagCommand('Freeze layer', () => frozen, (v) => (frozen = v), true, () => {}),
    );
    expect(frozen).toBe(true);
    stack.undo();
    expect(frozen).toBe(false);
    stack.redo();
    expect(frozen).toBe(true);
  });
});

describe('structural commands', () => {
  it('restores opaque snapshots in both directions', () => {
    // After a structural change the old parameter array describes nothing in
    // the new network, so the whole state is the only honest description.
    let state = { layers: [2, 3], params: [1, 2, 3] };
    const before = structuredClone(state);
    const after = { layers: [2, 5], params: [9, 9, 9, 9, 9] };
    const stack = new CommandStack();
    stack.execute(
      structuralCommand('Resize layer', before, after, (s) => {
        state = structuredClone(s);
      }),
    );
    expect(state.layers).toEqual([2, 5]);
    stack.undo();
    expect(state.layers).toEqual([2, 3]);
    expect(state.params).toEqual([1, 2, 3]);
  });
});
