"""keyword_rules.py - Tầng 0: luật từ khóa đọc từ keyword_rules.yaml (không dùng LLM/embedding)."""
import math, re, unicodedata
import yaml

def norm(s: str) -> str:
    """lower + bỏ dấu + đ->d + bỏ ký tự đặc biệt + gọn khoảng trắng."""
    s = unicodedata.normalize("NFD", (s or "").lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def light(s: str) -> str:
    """Chuẩn hóa NHẸ: chữ thường, giữ dấu tiếng Việt, bỏ ký tự đặc biệt (để phân biệt từ đồng âm)."""
    s = unicodedata.normalize("NFC", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s)).strip()

def has_diacritics(s: str) -> bool:
    s = (s or "").lower()
    return "đ" in s or any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s))

def _pad(s):               # thêm khoảng trắng hai đầu để khớp theo cụm từ nguyên vẹn
    return f" {s} "

class KeywordRules:
    STRENGTHS = ("decisive", "strong", "medium", "weak")

    def __init__(self, path="keyword_rules.yaml"):
        with open(path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        sc = self.cfg["scoring"]
        self.W, self.TM = sc["weights"], sc["title_multiplier"]
        self.NEG, self.MIN = sc["negative_penalty"], sc["min_score_to_fire"]
        self.CAP, self.BETA = sc["max_label_score"], sc["softmax_beta"]
        self.MARGIN, self.PRIOR_SCALE = sc["confusable_margin"], sc["organizer_prior_scale"]
        self.labels = {}
        for lab, spec in self.cfg["labels"].items():
            self.labels[lab] = {
                **{st: [norm(p) for p in spec.get(st, [])] for st in self.STRENGTHS},
                "negative": [norm(p) for p in spec.get("negative", [])],
                "combos": [{"all": [norm(p) for p in c["all"]], "weight": c["weight"]} for c in spec.get("combos", [])],
            }
        self.adjust = []
        for r in self.cfg.get("adjust_rules", []):
            self.adjust.append({
                "id": r["id"], "label": r["label"], "delta": r["delta"],
                "if_any": [norm(p) for p in r.get("if_any", [])],
                "if_any_2": [norm(p) for p in r.get("if_any_2", [])],
                "if_all": [norm(p) for p in r.get("if_all", [])],
                "if_none": [norm(p) for p in r.get("if_none", [])],
            })
        self.homographs = []
        for h in self.cfg.get("homographs", []):
            self.homographs.append({"stripped": norm(h["stripped"]), "weight": h["weight"],
                "variants": [{"text": light(v["text"]), "label": v["label"],
                              "context": [norm(c) for c in v["context_any"]]} for v in h["variants"]]})
        self.priors = [(p["match"], p["boosts"]) for p in self.cfg.get("organizer_priors", [])]
        self.confusable = {frozenset((a, b)): why for a, b, why in self.cfg.get("confusable_pairs", [])}
        self.attr = self.cfg["attributes"]

    @staticmethod
    def _in(padded, phrase):
        return _pad(phrase) in padded

    def score(self, title="", body="", organizer=""):
        t, b = _pad(norm(title)), _pad(norm(body))
        full = t + b
        org = norm(organizer)
        scores, hits = {}, {}
        for lab, spec in self.labels.items():
            s, h = 0.0, []
            for st in self.STRENGTHS:
                for ph in spec[st]:
                    if self._in(t, ph):   s += self.W[st] * self.TM; h.append((ph, st, "title"))
                    elif self._in(b, ph): s += self.W[st];           h.append((ph, st, "body"))
            for c in spec["combos"]:
                if all(self._in(full, p) for p in c["all"]):
                    s += self.W[c["weight"]]; h.append(("+".join(c["all"]), "combo:" + c["weight"], "text"))
            for ph in spec["negative"]:
                if self._in(full, ph): s -= self.NEG; h.append((ph, "negative", "text"))
            scores[lab], hits[lab] = s, h
        raw_light = _pad(light(title + " " + body))
        accented = has_diacritics(title + " " + body)
        for h in self.homographs:                               # từ đồng âm khi bỏ dấu
            for where, padded in (("title", t), ("body", b)):
                if not self._in(padded, h["stripped"]): continue
                if accented:
                    allowed = [v for v in h["variants"] if self._in(raw_light, v["text"])]
                else:
                    ctx = [v for v in h["variants"] if any(self._in(full, c) for c in v["context"])]
                    allowed = ctx if len(ctx) == 1 else h["variants"]
                for v in allowed:
                    mult = self.TM if where == "title" else 1.0
                    scores[v["label"]] += self.W[h["weight"]] * mult
                    hits[v["label"]].append((v["text"], "homograph:" + h["weight"], where))
                break                                           # mỗi cụm chỉ tính 1 lần (ưu tiên tiêu đề)
        for r in self.adjust:                                   # điều chỉnh chéo
            ok = (not r["if_any"] or any(self._in(full, p) for p in r["if_any"])) \
                 and (not r["if_any_2"] or any(self._in(full, p) for p in r["if_any_2"])) \
                 and all(self._in(full, p) for p in r["if_all"]) \
                 and not any(self._in(full, p) for p in r["if_none"])
            if ok and (r["if_any"] or r["if_all"]):
                scores[r["label"]] += r["delta"]; hits[r["label"]].append((r["id"], "adjust", f'{r["delta"]:+}'))
        for pats, boosts in self.priors:                        # tiên nghiệm theo đơn vị
            if any(re.search(p, org) for p in pats):
                for lab, w in boosts.items():
                    if lab in scores: scores[lab] += w * self.PRIOR_SCALE; hits[lab].append(("org", "prior", f"+{w}"))
        scores = {k: min(max(v, 0.0), self.CAP) for k, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top, second = ranked[0], ranked[1]
        fired = top[1] >= self.MIN
        probs = None
        if fired:
            z = {k: math.exp(self.BETA * v) for k, v in scores.items()}
            tot = sum(z.values()); probs = {k: v / tot for k, v in z.items()}
        decisive = fired and any(st == "decisive" for _, st, _ in hits[top[0]])
        confusable = False; why = None
        if fired and second[1] >= self.MIN and top[1] - second[1] < self.MARGIN:
            confusable = True; why = self.confusable.get(frozenset((top[0], second[0])), "hai nhãn có điểm gần nhau")
        status = "none" if not fired else ("review" if confusable else ("decisive" if decisive else "clear"))
        return dict(scores=scores, hits=hits, top=top[0] if fired else None, top2=[ranked[0][0], ranked[1][0]] if fired else [],
                    probs=probs, status=status, confusable_reason=why)

    # ---------------- thuộc tính suy ra ----------------
    def level(self, text, organizer=""):
        full = _pad(norm(text) + " " + norm(organizer))
        order = ["quoc_te", "quoc_gia", "thanh_pho", "dhqg", "truong", "khoa", "lop"]
        found = [lv for lv in order if any(self._in(full, norm(p)) for p in self.attr["level"][lv])]
        return found[0] if found else None, found

    def role(self, text):
        full = _pad(norm(text))
        roles = [r for r, ps in self.attr["role"].items() if any(self._in(full, norm(p)) for p in ps)]
        recruit = any(self._in(full, norm(c)) for c in self.attr["recruitment_cues"])
        return roles, bool(roles and recruit)

    def duration(self, text):
        t = norm(text)
        for m in re.finditer(self.attr["duration"]["unit_patterns"][0], t):
            n, unit = int(m.group(1)), m.group(2)
            if unit in ("tuan", "thang"): return "ge_1w"
            if unit == "ngay":
                if n >= 7: return "ge_1w"
                if n >= 2: return "ambiguous_lt_24h_or_mid"
                return "lt_24h"
            if unit == "gio": return "lt_24h"
        for p in self.attr["duration"]["priors"]:
            if self._in(_pad(t), norm(p["phrase"])): return p["result"] + "?"
        return None

    def explicit_points(self, text):
        t, ep = norm(text), self.attr["explicit_points"]
        if re.search(ep["phu_dinh"], t): return dict(mentions_drl=True, negated=True, points=None)
        m = re.search(ep["cong_diem"], t) or re.search(ep["so_diem"], t)
        mention = any(self._in(_pad(t), p) for p in ep["nhac_den_drl"])
        return dict(mentions_drl=mention or bool(m), negated=False, points=int(m.group(1)) if m else None)
