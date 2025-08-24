from pathlib import Path
import os
import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
) 
def main(cfg):
    from robobase.workspace import Workspace

    root_dir = Path.cwd()

    workspace = Workspace(cfg)
    # put cups: 'exp_local/pixel_act/bigym_put_cups_20241217133245/snapshots/17500_snapshot.pt'
    # sandwich remove: 'exp_local/pixel_act/bigym_sandwich_remove_20241217133401/snapshots/20000_snapshot.pt'
    snapshot = workspace.work_dir/'snapshots/best_snapshot.pt'
    assert snapshot.exists()
    print(f"resuming: {snapshot}")
    workspace.load_snapshot(snapshot)
    workspace.eval()


if __name__ == "__main__":
    main()
