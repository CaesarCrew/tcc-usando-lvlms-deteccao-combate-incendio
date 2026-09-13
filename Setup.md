As we all know, due to Large-Language models' high usage of computing resources it is essential to have the right set of tools to
make the most out of our models, these file will contain the documentation of the ones I used and how to set them up.

## Nvidia Drivers
First things first, most of the tools will depend in a good graphics card working correctly in order to reach peak efficiency,
therefore the first must-have tool is the correct nvidia drivers for your device.

The first step for the correct instalation is finding your GPU model and code name.
You can do it by running "lspci -v | grep -A10 VGA" or "" on linux, which will return :

In windows there are many methods, you can run the command "" on cmd or Powershell or taking a peak at "Performance" tab in Task
Manager (which can be open with the shortcut Ctrl+Shift+Esc) or yet by accessing Device Manager, in "Display Adapters".
With the card name in hands, you can procced with searching the equivalent code name to your card.
I highly recommend accessing [nouveau](https://nouveau.freedesktop.org/CodeNames.html), however, if your card is not in their list,
you can always visit [wikipedia](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units).

With that code, you can proceed and search the correct driver to your GPU.

Then, you can search it in the [nvidia website](https://www.nvidia.com/en-us/drivers/), specially if you're using Windows. If you're
a linux enjoyer however, my advice is to install it from your package manager or favorite specialized repository.
If you're using Arch linux specifically, I suggest that you use [AUR](https://aur.archlinux.org/), which may require the [guide](https://wiki.archlinux.org/title/Arch_User_Repository) if it's your first time using it.

## Environment
In order to maintain environments configured to not interfere which each model requirements, I highly recommend the use of a environment manager. I use [Anaconda](https://www.anaconda.com/docs/getting-started/installation) since I'm used to it, but you may be free to use simply [venv](https://docs.python.org/3/library/venv.html), or XXXXXX.

## Parallel Computing

[guide](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html)

### Arch Instalation Troubleshooting
If after installing and reboot Arch, you receive the message:

NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running

You should check your [initramfs](https://bbs.archlinux.org/viewtopic.php?id=141699) - specially if you have early start activated,
making sure to remove kms from the HOOKS and adding the nvidia modules first in the system initialization.
Also, ensure that [nouveau is blacklisted](https://bbs.archlinux.org/viewtopic.php?id=141699), it can be done either by creating "`/etc/modprobe.d/blacklist.conf`" file with the following contents:
blacklist nouveau

If that still doesn't work, it is wise to disable Simple Framebuffer your bootloader config, by adding "`initcall_blacklist=simplefb_init`"
to your kernel parameters.
### Ubuntu Instalation Troubleshooting
[This](https://forums.developer.nvidia.com/t/nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-make-sure-that-the-latest-nvidia-driver-is-installed-and-running/197141)
