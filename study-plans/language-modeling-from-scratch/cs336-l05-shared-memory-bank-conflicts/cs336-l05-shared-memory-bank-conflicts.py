import numpy as np

def bank_conflict_analysis(byte_addresses: np.ndarray, num_banks: int = 32, bank_width_bytes: int = 4) -> dict:
    """
    Returns a dict of int64 arrays: bank_ids and conflict_degree.
    """
    addr = np.asarray(byte_addresses, dtype=np.int64)

    # b_i = floor(a_i / w) mod B
    bank_ids = (addr // bank_width_bytes) % num_banks

    # conflict degree = number of distinct addresses requested from the lane's bank
    conflict_degree = np.zeros(addr.shape[0], dtype=np.int64)
    for b in np.unique(bank_ids):
        in_bank = bank_ids == b
        distinct = np.unique(addr[in_bank]).size
        conflict_degree[in_bank] = distinct

    return {
        "bank_ids": bank_ids.astype(np.int64),
        "conflict_degree": conflict_degree,
    }

# print(bank_conflict_analysis(np.array([0, 128, 4, 132])))