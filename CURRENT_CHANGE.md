## Changes

- Added the albatross inference backend, supporting batch inference while keeping API compatibility. The backend automatically handles batching. It works out of the box on Windows with RTX 30XX series and newer GPUs; select the option below in the client to get an immediate performance boost. Under concurrent workloads, 3060 and newer GPUs can typically reach 3000-10000 token/s inference speed for 3B and 7B scale models.

<img width="500" alt="Image" src="https://github.com/user-attachments/assets/aaa063fc-b4a9-4612-ab0b-9e3d2daf9dd3" />

- Added a batch generation button to the client, providing a friendlier batch generation preview interface.

https://github.com/user-attachments/assets/28aa789e-ad52-41bd-b8b5-0bcf8e9cd25f

- Bumped precompiled llama.cpp vulkan.

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
- Windows 7 Patches: https://github.com/josStorer/wails/releases/tag/v2.9.2x
