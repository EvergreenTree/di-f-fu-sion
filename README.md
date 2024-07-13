# di-f-fu-sion
Mechanics-inspired diffusion model structures and illustrative toy data notebooks for educational purposes.

## Train a Model
```
PYTHONPATH=. python maxdiffusion/models/train.py maxdiffusion/configs/base15.yml 
```

## Multi-Host Configurations

Please follow the awesome [TPU Starter](https://github.com/ayaka14732/tpu-starter).

<details>
<summary>
(Click to expand) For macos iTerm users, consider diagnosing all hosts with ssh using the following AppleScript. Then press `cmd+shift+I` to spread commands to each window.
</summary>

```osascript
tell application "iTerm"
	activate
	create window with default profile
	
	set sessionList to {current session of current window}
	
	tell current session of current window
		set newSession to (split vertically with default profile)
	end tell
	set beginning of sessionList to newSession
	
	repeat with i from 1 to 6
		tell current session of current window
			set newSession to (split horizontally with default profile)
		end tell
		set end of sessionList to newSession
	end repeat
	
	repeat with i from 0 to 7
		tell item (i + 1) of sessionList
			write text "gcloud compute tpus tpu-vm ssh ergodic-diffusion --worker=" & i
		end tell
	end repeat
end tell
```
</details>

<details>
<summary>
(Click to expand) Workaround to run jax in notebook on multi-host TPU pod: run `podrun -w -- python -c "import jax;jax.distributed.initialize()"` to initialize other nodes (however the process will halt after a while).
</summary>

```
from src.podrun import podrun

command = """import jax;jax.distributed.initialize()"""
podrun(command)
```

</details>

Run `ps -ef|grep python|grep USERNAME|grep -v grep|awk '{print $2}'|xargs sudo kill` on each node to clean processes if needed.

## Single-Host Demo

See `diffusion_lens.ipynb`