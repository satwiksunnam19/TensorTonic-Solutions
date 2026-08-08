def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns: list[int] containing token IDs after applying the ordered merge rules
    """
    seq = list(text.encode("utf-8"))
    for rule in merges:
        l, r, n = rule
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == (l, r):
                new_seq.append(n)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
    return seq


def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns: the Unicode string reconstructed from token IDs and vocabulary bytes
    """
    all_bytes = b""
    for token_id in ids:
        all_bytes += bytes(vocab[token_id])
    return all_bytes.decode("utf-8")