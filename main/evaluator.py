

def evaluate(type, golds, preds):
    assert len(golds) == len(preds)
    if type == "acc":
        correct = 0.0
        for _ in range(len(golds)):
            gold = golds[_]
            gold_str = " ".join(gold).strip()

            pred = preds[_]
            pred_str = " ".join(pred).strip()

            if gold_str == pred_str:
                correct += 1.0
        return correct/len(preds)