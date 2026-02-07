/**
 * UndoManager — Command pattern undo/redo stack.
 * Supports transform operations with max 50 steps.
 */

export class UndoManager {
  constructor(sceneManager, maxSteps = 50) {
    this.sceneManager = sceneManager;
    this.maxSteps = maxSteps;
    this.undoStack = [];
    this.redoStack = [];
  }

  push(command) {
    this.undoStack.push(command);
    if (this.undoStack.length > this.maxSteps) {
      this.undoStack.shift();
    }
    this.redoStack = [];
  }

  undo() {
    const cmd = this.undoStack.pop();
    if (!cmd) return;
    this._apply(cmd, cmd.before);
    this.redoStack.push(cmd);
  }

  redo() {
    const cmd = this.redoStack.pop();
    if (!cmd) return;
    this._apply(cmd, cmd.after);
    this.undoStack.push(cmd);
  }

  _apply(cmd, state) {
    if (cmd.type === 'transform') {
      this.sceneManager.setTransform(cmd.layerId, state);
    }
  }

  canUndo() { return this.undoStack.length > 0; }
  canRedo() { return this.redoStack.length > 0; }
}
