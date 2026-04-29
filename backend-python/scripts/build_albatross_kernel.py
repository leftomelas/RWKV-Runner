import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="sm80,sm86,sm89,sm90")
    parser.add_argument("--output-root", default="albatross/kernels")
    args = parser.parse_args()
    print(
        f"Requested Albatross kernel build for arch={args.arch} "
        f"output_root={args.output_root}"
    )
    print(
        "Build implementation should compile rwkv7_state_fwd_fp16 "
        "and update manifest.json."
    )


if __name__ == "__main__":
    main()
