# TPU Configurations
Please follow the awesome [TPU Starter](https://github.com/ayaka14732/tpu-starter).
<details>
<summary>
(Click to expand) Here's an example start-up script for a TPU VM.
</summary>
```
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch byobu nfs-common ffmpeg iotop iftop software-properties-common

sudo mkdir -p /nfs_share
sudo mount NFS_SERVER_IP:/nfs_share /nfs_share
ln -sf /nfs_share ~/nfs_share

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11-full python3.11-dev
python3.11 -m venv ~/venv

. ~/venv/bin/activate
echo ". ~/venv/bin/activate" >> .bashrc

pip install --upgrade pip wheel
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torch_xla[tpu] torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
pip install flax optax plotly matplotlib tensorflow-cpu tqdm jax-smi celluloid ipykernel ipython jupyter tensorflow-datasets tensorboardx jax-smi clu einops wandb huggingface_hub google-cloud-storage
```

To setup an nfs shared folder: (replace 172.21.12.0/24 with subnet IP range)
```
sudo apt-get install -y -qq nfs-kernel-server
sudo mkdir -p /nfs_share
sudo chown -R nobody:nogroup /nfs_share
sudo chmod 777 /nfs_share
ln -sf /nfs_share ~/nfs_share
sudo echo '/nfs_share  172.21.12.0/24(rw,sync,no_subtree_check)' > /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

</details>



<details>
<summary>
(Click to expand) For macos iTerm users, consider broadcasting commands to multiple windows using the following AppleScript (for a v4-32 TPU VM). Press `cmd+shift+I` to spread commands.
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
    
    repeat with i from 1 to 4
        tell current session of current window
            set newSession to (split horizontally with default profile)
        end tell
        set end of sessionList to newSession
    end repeat
    
    repeat with i from 0 to 3
        tell item (i + 1) of sessionList
            write text "gcloud compute tpus tpu-vm ssh TPU_NAME --worker=" & i
        end tell
    end repeat
end tell
```
</details>
