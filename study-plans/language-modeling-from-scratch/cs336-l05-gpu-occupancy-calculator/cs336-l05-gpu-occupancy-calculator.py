import math

def gpu_occupancy(threads_per_block, registers_per_thread, shared_mem_per_block,
                  max_threads_per_sm, max_warps_per_sm, max_blocks_per_sm,
                  max_registers_per_sm, max_shared_mem_per_sm, warp_size=32):
    # warps allocated per block, and the effective thread count they imply
    w = math.ceil(threads_per_block / warp_size)
    eff_threads = w * warp_size

    # cap on blocks from each of the five limits
    caps = [
        max_threads_per_sm // eff_threads,
        max_warps_per_sm // w,
        max_blocks_per_sm,
    ]

    # zero register / shared-memory use does not constrain the block count
    regs_per_block = registers_per_thread * eff_threads
    if regs_per_block > 0:
        caps.append(max_registers_per_sm // regs_per_block)
    if shared_mem_per_block > 0:
        caps.append(max_shared_mem_per_sm // shared_mem_per_block)

    b = min(caps)
    resident_warps = b * w
    occupancy = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm": int(b),
        "resident_warps": int(resident_warps),
        "occupancy": float(occupancy),
    }

print(gpu_occupancy(100, 64, 2048, 1024, 32, 16, 32768, 49152, 32))