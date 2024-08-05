Mechanics-inspired diffusion model structures and illustrative toy data notebooks for educational purposes. Based on [maxdiffusion](https://github.com/google/maxdiffusion) and [ddpm-flax](https://github.com/yiyixuxu/denoising-diffusion-flax)

## Train a Model
```
python maxdiffusion/main.py --config=maxdiffusion/configs/cifar10.py
```

## Single-Host TPU Demo

See [diffusion_lens.ipynb](diffusion_lens.ipynb)

## TPU Configurations

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

Run `pkill -f process_name` or `ps -ef|grep python|grep USERNAME|grep -v grep|awk '{print $2}'|xargs sudo kill` on each node to clean hidden processes using TPU if needed.