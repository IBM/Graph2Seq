import configure as conf

def decode_text(pred_idx, idx_word):
    # if conf.decoder_type != "beam":
    #     pred_idx = pred_idx[0]
    # else:
    #     pred_idx = np.transpose(pred_idx)
    #     pred_idx = pred_idx[0]

    text = ""
    for __ in range(len(pred_idx)):
        ids = pred_idx[__]
        if isinstance(ids, list):
            for id in ids:
                if id == 2:
                    break
                if id != 0:
                    text += idx_word[id] + " "
                else:
                    text += conf.GO + " "
        else:
            if ids == 2:
                break
            if ids != 0:
                text += idx_word[ids] + " "
            else:
                text += conf.GO + " "
    return text