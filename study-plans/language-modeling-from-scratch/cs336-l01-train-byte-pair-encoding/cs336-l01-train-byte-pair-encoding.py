from collections import Counter

def train_bpe(corpus, vocab_size):
    encoded = [list(s.encode("utf-8")) for s in corpus]

    id_to_bytes = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges = []
    vocab = []

    while next_id < vocab_size:
        adj_pairs = []
        for seq in encoded:
            for j in range(len(seq)-1):
                adj_pairs.append((seq[j], seq[j+1]))

        if not adj_pairs:
            break

        freq_counter = Counter(adj_pairs)
        freq_dict = dict(freq_counter)
        freq_list = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)

        best_count = freq_list[0][1]
        tied_pairs = []
        for i in range(len(freq_list)):
            if freq_list[i][1] == best_count:
                tied_pairs.append(freq_list[i][0])

        best_pair = max(tied_pairs, key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]]))
        
        new_encoded = []
        for seq in encoded:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == best_pair:
                    new_seq.append(next_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_encoded.append(new_seq)
        encoded = new_encoded

        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        id_to_bytes[next_id] = new_bytes
        vocab.append([next_id, list(new_bytes)])
        merges.append([best_pair[0], best_pair[1], next_id])

        next_id += 1

    return {"vocab": vocab, "merges": merges}
