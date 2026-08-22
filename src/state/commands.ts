/*
 * The undo/redo command stack.
 *
 * Spec §6.5: "Undo/redo over an explicit command stack covering all edits and
 * architecture changes."
 *
 * Two kinds of command, because the two kinds of edit have very different
 * costs:
 *
 *   - A SCALAR edit (one weight, one bias) stores just the index and the two
 *     values. Snapshotting the whole parameter array to record a single dragged
 *     weight would allocate a thousand floats per undo entry.
 *
 *   - A STRUCTURAL edit (adding a layer, resizing one, switching a dataset)
 *     changes the shape of the parameter array itself, so there is no index to
 *     record and the full before/after state is the only honest description.
 *
 * A drag produces ONE command, not one per pointer move: `beginScrub` captures
 * the value at mousedown and `commitScrub` pushes a single entry at mouseup.
 * Otherwise a single scrub would bury the previous edit under a hundred undos.
 */

export interface Command {
  /** Shown in the UI, so undo can say what it will undo. */
  readonly label: string;
  apply(): void;
  revert(): void;
  /**
   * Merge with the previous command if they describe the same continuous
   * gesture, so repeated nudges of one weight collapse into a single undo.
   */
  mergeWith?(previous: Command): Command | null;
}

export const MAX_UNDO_DEPTH = 200;

export class CommandStack {
  private readonly done: Command[] = [];
  private readonly undone: Command[] = [];

  get canUndo(): boolean {
    return this.done.length > 0;
  }

  get canRedo(): boolean {
    return this.undone.length > 0;
  }

  get undoLabel(): string | null {
    return this.done[this.done.length - 1]?.label ?? null;
  }

  get redoLabel(): string | null {
    return this.undone[this.undone.length - 1]?.label ?? null;
  }

  get depth(): number {
    return this.done.length;
  }

  /** Run a command and record it. Clears the redo stack, as any editor does. */
  execute(command: Command): void {
    command.apply();
    const previous = this.done[this.done.length - 1];
    if (previous !== undefined && command.mergeWith !== undefined) {
      const merged = command.mergeWith(previous);
      if (merged !== null) {
        this.done[this.done.length - 1] = merged;
        this.undone.length = 0;
        return;
      }
    }
    this.done.push(command);
    // Oldest entries fall off the bottom rather than growing without bound.
    if (this.done.length > MAX_UNDO_DEPTH) this.done.shift();
    this.undone.length = 0;
  }

  /** Record a command whose effect has ALREADY been applied. */
  record(command: Command): void {
    this.done.push(command);
    if (this.done.length > MAX_UNDO_DEPTH) this.done.shift();
    this.undone.length = 0;
  }

  undo(): Command | null {
    const command = this.done.pop();
    if (command === undefined) return null;
    command.revert();
    this.undone.push(command);
    return command;
  }

  redo(): Command | null {
    const command = this.undone.pop();
    if (command === undefined) return null;
    command.apply();
    this.done.push(command);
    return command;
  }

  clear(): void {
    this.done.length = 0;
    this.undone.length = 0;
  }
}

/** One scalar parameter, addressed by its index in the flat parameter array. */
export function setParameterCommand(
  label: string,
  params: Float64Array,
  index: number,
  before: number,
  after: number,
  onChange: () => void,
): Command {
  return {
    label,
    apply(): void {
      params[index] = after;
      onChange();
    },
    revert(): void {
      params[index] = before;
      onChange();
    },
    mergeWith(previous: Command): Command | null {
      // Consecutive edits to the SAME parameter collapse, so holding an arrow
      // key does not fill the undo stack with single-step nudges.
      const p = previous as Partial<{ parameterIndex: number; beforeValue: number }>;
      if (p.parameterIndex !== index) return null;
      return setParameterCommand(label, params, index, p.beforeValue as number, after, onChange);
    },
    parameterIndex: index,
    beforeValue: before,
  } as Command & { parameterIndex: number; beforeValue: number };
}

/** A boolean flag on a layer, such as frozen or ablated. */
export function setFlagCommand(
  label: string,
  read: () => boolean,
  write: (value: boolean) => void,
  after: boolean,
  onChange: () => void,
): Command {
  const before = read();
  return {
    label,
    apply(): void {
      write(after);
      onChange();
    },
    revert(): void {
      write(before);
      onChange();
    },
  };
}

/**
 * A change that alters the SHAPE of the network.
 *
 * The before and after are opaque snapshots, because after a structural change
 * the old parameter array no longer describes anything in the new network. The
 * caller supplies serialize/restore so this module stays free of engine types.
 */
export function structuralCommand<T>(
  label: string,
  before: T,
  after: T,
  restore: (state: T) => void,
): Command {
  return {
    label,
    apply(): void {
      restore(after);
    },
    revert(): void {
      restore(before);
    },
  };
}
