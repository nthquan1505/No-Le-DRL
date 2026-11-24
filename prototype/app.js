/* No-Le-DRL Module 3 Interactive Prototype Client */

const GROUP_NAMES = {
  C1: "Học tập & NCKH",
  C2: "Chấp hành nội quy",
  C3: "CT-XH, VH-VN, TDTT",
  C4: "Phát triển cộng đồng",
  C5: "Nội dung khuyến khích",
};

const QUICK_SAMPLES = {
  ctf: {
    title: "Cuộc thi An toàn thông tin HCMUS CTF 2026",
    description: "Cuộc thi học thuật cấp Khoa dành cho sinh viên đam mê an toàn thông tin và lập trình.",
    organizer: "Khoa CNTT",
    role: "participant",
  },
  ctf_btc: {
    title: "Cuộc thi Coding Challenge 2026",
    description: "Tuyển Ban tổ chức hỗ trợ điều phối và chấm bài cho cuộc thi học thuật cấp Khoa.",
    organizer: "ĐK CNTT",
    role: "organizer",
  },
  seminar: {
    title: "Seminar Học thuật về Nghiên cứu AI và Machine Learning",
    description: "Buổi chia sẻ chuyên đề nghiên cứu khoa học sinh viên và định hướng khóa luận tốt nghiệp.",
    organizer: "Khoa CNTT",
    role: "participant",
  },
  volunteer: {
    title: "Chiến dịch Tình nguyện Mùa Hè Xanh 3 ngày 2 đêm",
    description: "Hoạt động thiện nguyện nấu ăn và phát quà cho trẻ em vùng khó khăn.",
    organizer: "Đoàn Thanh niên - Hội Sinh viên HCMUS",
    role: "participant",
  },
};

const state = {
  bootstrap: null,
  recommendationsBundle: null,
  schedulePlan: null,
  profile: null,
  rubricCatalog: [],
  feedbackLog: [],
  clsIndexSummary: null,
  modalEventId: null,
  searchQuery: "",
};

const $ = (sel) => document.querySelector(sel);

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    el.hidden = true;
  }, 3600);
}

function formatDateVi(isoStr) {
  if (!isoStr) return "Chưa rõ thời gian";
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return isoStr;
  return d.toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatFeeVi(feeVnd, feeRaw) {
  if (feeVnd === 0) return "Miễn phí";
  if (typeof feeVnd === "number") return `${feeVnd.toLocaleString("vi-VN")}đ`;
  return feeRaw ? `Chưa chuẩn hóa` : "Không ghi phí";
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/* =========================================================================
   RENDERERS
   ========================================================================= */

function renderHeaderStatus() {
  const st = state.bootstrap.embedder_status;
  const prof = state.profile;
  $("#studentChip").textContent = `${prof.name} · ${prof.cohort} · ${prof.class_id}`;

  const banner = $("#embedderBanner");
  const clsBadge = $("#clsModelBadge");
  if (st.is_e5_active) {
    banner.className = "status-ribbon";
    banner.innerHTML = `
      <span class="badge badge-teal mono">P2.1 ACTIVE</span>
      <span>Model Embedding: <strong class="mono">${escapeHtml(st.active_model)}</strong> (${st.dimension}d · L2 · device=${escapeHtml(st.device)})</span>
    `;
    if (clsBadge) {
      clsBadge.className = "badge badge-teal mono";
      clsBadge.textContent = `e5-small · ${st.dimension}d`;
    }
  } else {
    banner.className = "status-ribbon fallback";
    banner.innerHTML = `
      <span class="badge badge-orange mono">P2.1 E5-SMALL STANDBY</span>
      <span>Mục tiêu: <strong class="mono">${escapeHtml(st.requested_model)}</strong> (384d) · Hiện đang chạy fallback <code class="mono">${escapeHtml(st.active_model)}</code> vì môi trường chưa cài <code class="mono">sentence-transformers</code>. Chạy <code class="mono">pip install sentence-transformers torch</code> để kích hoạt e5-small thật.</span>
    `;
    if (clsBadge) {
      clsBadge.className = "badge badge-orange mono";
      clsBadge.textContent = `e5-small (standby) · ${st.dimension}d`;
    }
  }
}

function renderProgressStrip() {
  const bundle = state.recommendationsBundle;
  const led = bundle.ledger_summary;
  $("#totalScoreVal").textContent = `${led.total_score}đ`;
  $("#targetScoreVal").textContent = `${led.target_total}`;
  $("#kpiTotalEvents").textContent = bundle.stats.total_cached_events;
  $("#kpiEligible").textContent = bundle.stats.eligible_count;
  $("#kpiDropped").textContent = bundle.stats.dropped_count;
  $("#kpiClsSize").textContent = `${state.clsIndexSummary.total_samples}`;
  $("#refNowLabel").textContent = formatDateVi(bundle.reference_now);

  const capsHtml = ["C1", "C2", "C3", "C4", "C5"]
    .map((grp) => {
      const cur = led.capped_by_group[grp] ?? 0;
      const cap = led.group_caps[grp] ?? 25;
      const gap = led.target_gap_by_group[grp] ?? 0;
      const pct = Math.min(100, Math.round((cur / cap) * 100));
      const isFull = cur >= cap;
      return `
        <div class="cap-item">
          <div class="cap-top">
            <span><span class="mono">${grp}</span> · ${GROUP_NAMES[grp]}</span>
            <span class="mono">${cur}/${cap}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${isFull ? "full" : ""}" style="width:${pct}%"></div>
          </div>
          <div class="cap-sub">${isFull ? "Đã chạm trần nhóm" : gap > 0 ? `Còn thiếu ${gap}đ mục tiêu` : "Đã đạt mục tiêu nhóm"}</div>
        </div>
      `;
    })
    .join("");
  $("#groupCapsGrid").innerHTML = capsHtml;
}

function renderWeightControls() {
  const presets = state.bootstrap.weight_presets;
  const activePreset = state.profile.preset;
  $("#presetButtons").innerHTML = Object.entries(presets)
    .map(
      ([key, info]) => `
        <button class="preset-btn ${activePreset === key ? "active" : ""}" data-preset="${key}">
          ${escapeHtml(info.label)}
        </button>
      `
    )
    .join("");

  const w = state.recommendationsBundle.weights_applied;
  const setSlider = (id, labelId, val) => {
    const pct = Math.round(val * 100);
    $(id).value = pct;
    $(labelId).textContent = `${pct}%`;
  };
  setSlider("#w1Slider", "#w1Val", w.w1_gain);
  setSlider("#w2Slider", "#w2Val", w.w2_interest);
  setSlider("#w3Slider", "#w3Val", w.w3_fit);
  setSlider("#w4Slider", "#w4Val", w.w4_urgency);

  $("#toggleConflicts").checked = Boolean(state.profile.hide_schedule_conflicts);
  $("#toggleCap").checked = Boolean(state.profile.hide_group_cap_reached);
  $("#toggleExamWeek").checked = Boolean(state.profile.exam_week_mode);
}

function renderReminders() {
  const list = state.recommendationsBundle.reminders || [];
  if (!list.length) {
    $("#remindersList").innerHTML = `<p class="helper-text">Chưa có sự kiện nào sắp hết hạn hoặc được lưu.</p>`;
    return;
  }
  $("#remindersList").innerHTML = list
    .map(
      (r) => `
        <div class="reminder-item">
          <div class="reminder-title">${escapeHtml(r.title)}</div>
          <div>
            <span class="mono">${escapeHtml(r.rubric_item_id)}</span> · +${r.effective_points}đ ·
            ${r.deadline_alert ? escapeHtml(r.deadline_alert) : "Sự kiện bạn đã lưu"}
          </div>
        </div>
      `
    )
    .join("");
}

function renderRecommendations() {
  const q = state.searchQuery.trim().toLowerCase();
  const cards = (state.recommendationsBundle.recommendations || []).filter((c) => {
    if (!q) return true;
    return (
      c.title.toLowerCase().includes(q) ||
      c.organizer.toLowerCase().includes(q) ||
      c.rubric_item_id.toLowerCase().includes(q) ||
      c.parent_group_id.toLowerCase().includes(q)
    );
  });

  if (!cards.length) {
    $("#recommendationsGrid").innerHTML = `
      <div class="card" style="grid-column:1/-1">
        <p class="helper-text">Không có sự kiện phù hợp với bộ lọc hiện tại. Thử nới lỏng ràng buộc ở cột trái hoặc kiểm tra mục "Tại sao không gợi ý?" bên dưới.</p>
      </div>
    `;
  } else {
    $("#recommendationsGrid").innerHTML = cards
      .map((c) => {
        const bd = c.score_breakdown;
        const pct = (v) => Math.max(0, Math.min(100, Math.round(v * 100)));
        return `
          <article class="rec-card ${c.is_exploration_slot ? "exploration" : ""}">
            <div class="rec-top">
              <div class="rec-badges">
                <span class="badge badge-teal mono">${escapeHtml(c.rubric_item_id)} · ${escapeHtml(c.parent_group_id)}</span>
                <span class="badge badge-teal">+${c.effective_points}đ dự kiến</span>
                ${c.is_exploration_slot ? `<span class="badge badge-orange">Slot khám phá (P7)</span>` : ""}
                ${c.deadline_alert ? `<span class="badge badge-danger">${escapeHtml(c.deadline_alert)}</span>` : ""}
                ${c.is_label_overridden ? `<span class="badge badge-muted">User đã sửa nhãn</span>` : ""}
              </div>
              <div class="score-pill mono">Score ${c.score_total.toFixed(2)}</div>
            </div>

            <h3 class="rec-title">${escapeHtml(c.title)}</h3>

            <div class="rec-meta">
              <span><strong>Đơn vị:</strong> ${escapeHtml(c.organizer || "HCMUS")}</span>
              <span><strong>Thời gian:</strong> ${escapeHtml(formatDateVi(c.start_datetime))}</span>
              <span><strong>Hình thức:</strong> ${escapeHtml(c.online_mode || "chưa rõ")}</span>
              <span><strong>Phí:</strong> ${escapeHtml(formatFeeVi(c.fee_vnd, c.fee_raw))}</span>
            </div>

            <p class="rec-desc">${escapeHtml(c.description || "Xem chi tiết bài đăng sự kiện từ nguồn chính thức.")}</p>

            <div class="breakdown-grid" aria-label="Điểm thành phần P6">
              <div class="bd-cell">
                <div class="bd-head"><span>Gain</span><strong class="mono">${pct(bd.gain)}%</strong></div>
                <div class="bd-bar"><span style="width:${pct(bd.gain)}%"></span></div>
              </div>
              <div class="bd-cell">
                <div class="bd-head"><span>Interest</span><strong class="mono">${pct(bd.interest)}%</strong></div>
                <div class="bd-bar"><span style="width:${pct(bd.interest)}%"></span></div>
              </div>
              <div class="bd-cell">
                <div class="bd-head"><span>Fit</span><strong class="mono">${pct(bd.fit)}%</strong></div>
                <div class="bd-bar"><span style="width:${pct(bd.fit)}%"></span></div>
              </div>
              <div class="bd-cell">
                <div class="bd-head"><span>Urgency</span><strong class="mono">${pct(bd.urgency)}%</strong></div>
                <div class="bd-bar"><span style="width:${pct(bd.urgency)}%"></span></div>
              </div>
            </div>

            <div class="reasons-box">
              <ul>
                ${c.top_reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
              </ul>
            </div>

            <div class="rec-actions">
              <button class="btn ${c.is_saved ? "btn-saved" : "btn-ghost"}" data-action="save" data-id="${escapeHtml(c.event_id)}">
                ${c.is_saved ? "Đã lưu" : "Lưu"}
              </button>
              <button class="btn btn-primary" data-action="joined" data-id="${escapeHtml(c.event_id)}">
                Đã tham gia (+${c.effective_points}đ)
              </button>
              <button class="btn btn-ghost" data-action="edit_label" data-id="${escapeHtml(c.event_id)}" data-current="${escapeHtml(c.rubric_item_id)}">
                Sửa nhãn
              </button>
              <button class="btn btn-ghost" data-action="dismiss" data-id="${escapeHtml(c.event_id)}">
                Bỏ qua
              </button>
            </div>
          </article>
        `;
      })
      .join("");
  }

  const dropped = state.recommendationsBundle.dropped_events || [];
  $("#droppedSummaryCount").textContent = state.recommendationsBundle.stats.dropped_count;
  $("#droppedList").innerHTML = dropped
    .map(
      (d) => `
        <div class="dropped-row">
          <div>
            <strong>${escapeHtml(d.title)}</strong>
            <span class="mono text-muted"> (${escapeHtml(d.rubric_item_id || "?")} · ${escapeHtml(formatDateVi(d.start_datetime))})</span>
          </div>
          <div class="rec-badges">
            ${d.drop_reasons.map((r) => `<span class="badge badge-muted">${escapeHtml(r.label)}</span>`).join("")}
          </div>
        </div>
      `
    )
    .join("");
}

function renderPlanner() {
  const plan = state.schedulePlan;
  if (!plan) return;
  $("#plannerSolverNote").textContent = plan.solver_note;
  $("#plannerGoalRange").value = plan.target_total;
  $("#plannerGoalVal").textContent = `${plan.target_total}đ`;
  $("#planCurrentScore").textContent = `${plan.current_total}đ`;
  $("#planProjectedScore").textContent = `${plan.projected_total}đ`;
  $("#planEventCount").textContent = `${plan.planned_events.length}`;

  if (!plan.planned_events.length) {
    $("#plannedEventsList").innerHTML = `<p class="helper-text">Bạn đã đạt mục tiêu hoặc không còn sự kiện phù hợp để thêm.</p>`;
  } else {
    $("#plannedEventsList").innerHTML = plan.planned_events
      .map(
        (ev, idx) => `
          <div class="planned-row">
            <div>
              <div class="rec-badges" style="margin-bottom:5px">
                <span class="badge badge-teal mono">#${idx + 1} · ${escapeHtml(ev.rubric_item_id)} (${escapeHtml(ev.parent_group_id)})</span>
                <span class="badge badge-muted">${escapeHtml(ev.online_mode || "offline")}</span>
              </div>
              <div style="font-weight:700;margin-bottom:3px">${escapeHtml(ev.title)}</div>
              <div class="helper-text" style="margin:0">${escapeHtml(formatDateVi(ev.start_datetime))} · ${escapeHtml(ev.organizer || "HCMUS")}</div>
            </div>
            <div class="badge badge-teal mono" style="font-size:13px">+${ev.planned_gain}đ</div>
          </div>
        `
      )
      .join("");
  }

  $("#timetableList").innerHTML = (plan.weekly_timetable || [])
    .map(
      (slot) => `
        <div class="timetable-row">
          <div>
            <strong>${escapeHtml(slot.day_label)}</strong> · ${escapeHtml(slot.course)}
            <div class="helper-text" style="margin:0">${escapeHtml(slot.campus)}</div>
          </div>
          <span class="mono">${slot.start_hour.toFixed(1)}h – ${slot.end_hour.toFixed(1)}h</span>
        </div>
      `
    )
    .join("");

  $("#planGroupBars").innerHTML = ["C1", "C2", "C3", "C4", "C5"]
    .map((grp) => {
      const cur = plan.current_by_group[grp] ?? 0;
      const proj = plan.projected_by_group[grp] ?? cur;
      const cap = plan.group_caps[grp] ?? 25;
      const pct = Math.min(100, Math.round((proj / cap) * 100));
      return `
        <div>
          <div class="cap-top">
            <span><span class="mono">${grp}</span> · ${GROUP_NAMES[grp]}</span>
            <span class="mono">${cur} → <strong>${proj}</strong> / ${cap}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </div>
      `;
    })
    .join("");
}

function renderClassifyTrace(res) {
  const cls = res.classification;
  const m6 = res.module6_credit_estimate;
  const s1Entries = Object.entries(cls.stage1_scores || {}).sort((a, b) => b[1] - a[1]);
  const s2Entries = cls.top_k_items_named || [];

  $("#classifyResultPane").innerHTML = `
    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 1 · P2.1 Embedding (${escapeHtml(res.embedder_status.active_model)})</h2>
        <span class="mono tag-soft">${res.vector_dimension}d · ||v||₂ = ${res.vector_l2_norm} · ${res.elapsed_ms}ms</span>
      </div>
      <div class="query-preview mono">${escapeHtml(res.formatted_query_text)}</div>
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 2 · P2.2 Stage 1 — Coarse Parent Groups (C1–C5)</h2>
        <span class="badge badge-teal">Đã chọn Top-${res.knn_params_used.stage1_top_k}: ${escapeHtml(cls.selected_candidate_groups.join(", "))}</span>
      </div>
      <p class="helper-text">Stage 1 chỉ chọn nhóm cha ứng viên — chưa kết luận mục nhỏ (D-2). Điểm dưới đây là raw KNN score, không phải xác suất đã calibrate.</p>
      ${s1Entries
        .map(([grp, sc]) => {
          const isSel = cls.selected_candidate_groups.includes(grp);
          const pct = Math.round(Math.max(0, Math.min(1, sc)) * 100);
          return `
            <div class="score-bar-row">
              <span><strong class="mono">${grp}</strong> ${isSel ? '<span class="badge badge-teal">Top-k</span>' : ""}</span>
              <div class="bar-track"><div class="bar-fill ${isSel ? "" : "full"}" style="width:${pct}%;${isSel ? "" : "background:#94A3B8"}"></div></div>
              <span class="mono" style="text-align:right">${sc.toFixed(4)}</span>
            </div>
          `;
        })
        .join("")}
    </div>

    <div class="trace-step highlight">
      <div class="card-title-row">
        <h2 class="card-title">Bước 3 · P2.3 Stage 2 — Multi-Group Union KNN</h2>
        <span class="badge badge-orange mono">status: ${escapeHtml(cls.status)}</span>
      </div>
      <p class="helper-text">
        Đã hợp nhất <strong>${res.stage2_union_pool_size} mẫu</strong> thuộc các nhóm <strong class="mono">{${escapeHtml(cls.selected_candidate_groups.join(", "))}}</strong>.
        Kết quả dự đoán: <strong class="mono text-teal">${escapeHtml(cls.rubric_item_id || "None")} (${escapeHtml(cls.parent_group_id || "-")})</strong> — ${escapeHtml(cls.rubric_item_name || "")}.
      </p>
      ${s2Entries
        .map((row, idx) => {
          const pct = Math.round(Math.max(0, Math.min(1, row.stage2_score)) * 100);
          return `
            <div class="score-bar-row">
              <span><strong class="mono">${escapeHtml(row.rubric_item_id)}</strong> <span class="text-muted">(${escapeHtml(row.parent_group_id)})</span> ${idx === 0 ? '<span class="badge badge-teal">#1</span>' : ""}</span>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
              <span class="mono" style="text-align:right">${row.stage2_score.toFixed(4)}</span>
            </div>
          `;
        })
        .join("")}
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 4 · MODULE 6 — Credit Calc &amp; Evaluation (Tách biệt theo D-3)</h2>
        <span class="badge ${m6.is_credit_eligible ? "badge-teal" : "badge-muted"}">${m6.is_credit_eligible ? "Đủ điều kiện cộng điểm" : "Không cộng điểm / Đã chạm trần"}</span>
      </div>
      <div class="kpi-strip">
        <div class="kpi-box">
          <div class="kpi-label">Mã mục áp dụng</div>
          <div class="kpi-val mono">${escapeHtml(m6.rubric_item_id || "-")} (${escapeHtml(m6.group_id || "-")})</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Điểm quy chế</div>
          <div class="kpi-val mono">+${m6.expected_points}đ</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Điểm thực nhận (sau trần)</div>
          <div class="kpi-val mono text-teal">+${m6.effective_points}đ</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Nguồn tính</div>
          <div class="kpi-val mono">${escapeHtml(m6.source)}</div>
        </div>
      </div>
      ${
        m6.warning_flags && m6.warning_flags.length
          ? `<div style="margin-top:10px" class="rec-badges">${m6.warning_flags.map((f) => `<span class="badge badge-muted mono">${escapeHtml(f)}</span>`).join("")}</div>`
          : ""
      }
    </div>
  `;
}

function renderProfileAndFeedback() {
  const chips = state.bootstrap.interest_chips || [];
  const selected = new Set(state.profile.selected_chips || []);
  $("#interestChipsGrid").innerHTML = chips
    .map(
      (c) => `
        <button type="button" class="chip-btn ${selected.has(c.id) ? "selected" : ""}" data-chip="${escapeHtml(c.id)}">
          ${escapeHtml(c.label)}
        </button>
      `
    )
    .join("");

  $("#profCohort").value = state.profile.cohort || "";
  $("#profFaculty").value = state.profile.faculty || "";
  $("#profBudget").value = state.profile.max_budget_vnd ?? 50000;

  $("#feedbackCountBadge").textContent = `${state.feedbackLog.length} sự kiện`;
  if (!state.feedbackLog.length) {
    $("#feedbackTimeline").innerHTML = `<p class="helper-text">Chưa có tương tác nào. Hãy thử bấm <strong>Lưu</strong>, <strong>Đã tham gia</strong>, <strong>Bỏ qua</strong> hoặc <strong>Sửa nhãn</strong> ở tab Gợi ý ĐRL.</p>`;
  } else {
    $("#feedbackTimeline").innerHTML = state.feedbackLog
      .map(
        (f) => `
          <div class="feedback-item">
            <div class="card-title-row" style="margin-bottom:4px">
              <span class="badge badge-teal mono">${escapeHtml(f.action)}</span>
              <span class="mono text-muted" style="font-size:11.5px">${escapeHtml(f.timestamp)}</span>
            </div>
            <div style="font-weight:700">${escapeHtml(f.event_title)}</div>
            <div class="helper-text" style="margin:3px 0 0">${escapeHtml(f.detail || "")}</div>
          </div>
        `
      )
      .join("");
  }
}

function renderAll() {
  renderHeaderStatus();
  renderProgressStrip();
  renderWeightControls();
  renderReminders();
  renderRecommendations();
  renderPlanner();
  renderProfileAndFeedback();
}

/* =========================================================================
   EVENT BINDINGS
   ========================================================================= */

function openLabelModal(eventId, currentItemId) {
  state.modalEventId = eventId;
  const card = state.recommendationsBundle.recommendations.find((c) => c.event_id === eventId);
  $("#modalEventName").textContent = card ? card.title : eventId;
  const select = $("#modalLabelSelect");
  select.innerHTML = state.rubricCatalog
    .filter((item) => item.parent_group_id !== "P")
    .map(
      (item) => `
        <option value="${escapeHtml(item.rubric_item_id)}" ${item.rubric_item_id === currentItemId ? "selected" : ""}>
          ${escapeHtml(item.rubric_item_id)} (${escapeHtml(item.parent_group_id)}) — ${escapeHtml(item.name)}
        </option>
      `
    )
    .join("");
  $("#labelModal").hidden = false;
}

function closeLabelModal() {
  $("#labelModal").hidden = true;
  state.modalEventId = null;
}

async function runLiveClassify() {
  const payload = {
    title: $("#clsTitle").value,
    description: $("#clsDesc").value,
    organizer: $("#clsOrg").value,
    role: $("#clsRole").value,
    stage1_top_k: Number($("#paramTopK").value || 2),
    stage1_neighbors: Number($("#paramK1").value || 5),
    stage2_neighbors: Number($("#paramK2").value || 3),
  };
  const res = await apiPost("/api/classify", payload);
  renderClassifyTrace(res);
}

function setupListeners() {
  // Navigation tabs
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".nav-tab").forEach((b) => {
        const active = b.dataset.tab === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.id === `tab-${target}`);
      });
    });
  });

  // Search filter
  $("#feedSearch").addEventListener("input", (e) => {
    state.searchQuery = e.target.value;
    renderRecommendations();
  });

  // Weight presets
  $("#presetButtons").addEventListener("click", async (e) => {
    const btn = e.target.closest(".preset-btn");
    if (!btn) return;
    const res = await apiPost("/api/profile", { preset: btn.dataset.preset });
    state.profile = res.profile;
    state.recommendationsBundle = res.recommendations;
    renderAll();
    showToast(`Đã chuyển sang chế độ trọng số: ${btn.textContent.trim()}`);
  });

  // Weight sliders
  let sliderTimer = null;
  ["#w1Slider", "#w2Slider", "#w3Slider", "#w4Slider"].forEach((sel) => {
    $(sel).addEventListener("input", () => {
      clearTimeout(sliderTimer);
      sliderTimer = setTimeout(async () => {
        const weights = {
          w1_gain: Number($("#w1Slider").value) / 100,
          w2_interest: Number($("#w2Slider").value) / 100,
          w3_fit: Number($("#w3Slider").value) / 100,
          w4_urgency: Number($("#w4Slider").value) / 100,
        };
        const res = await apiPost("/api/profile", { weights });
        state.profile = res.profile;
        state.recommendationsBundle = res.recommendations;
        renderAll();
      }, 140);
    });
  });

  // P5 toggles
  const syncToggles = async () => {
    const res = await apiPost("/api/profile", {
      hide_schedule_conflicts: $("#toggleConflicts").checked,
      hide_group_cap_reached: $("#toggleCap").checked,
      exam_week_mode: $("#toggleExamWeek").checked,
    });
    state.profile = res.profile;
    state.recommendationsBundle = res.recommendations;
    state.schedulePlan = await apiGet("/api/plan");
    renderAll();
  };
  $("#toggleConflicts").addEventListener("change", syncToggles);
  $("#toggleCap").addEventListener("change", syncToggles);
  $("#toggleExamWeek").addEventListener("change", syncToggles);

  // Card action buttons (Save, Joined, Dismiss, Edit Label)
  $("#recommendationsGrid").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const eventId = btn.dataset.id;
    if (action === "edit_label") {
      openLabelModal(eventId, btn.dataset.current);
      return;
    }
    const res = await apiPost("/api/feedback", { action, event_id: eventId });
    state.feedbackLog = res.feedback_log;
    state.clsIndexSummary = res.cls_index_summary;
    state.recommendationsBundle = res.recommendations;
    state.schedulePlan = await apiGet("/api/plan");
    renderAll();
    showToast(res.entry.detail);
  });

  // Modal listeners
  $("#modalClose").addEventListener("click", closeLabelModal);
  $("#modalCancel").addEventListener("click", closeLabelModal);
  $("#modalConfirm").addEventListener("click", async () => {
    if (!state.modalEventId) return;
    const correctedItem = $("#modalLabelSelect").value;
    const res = await apiPost("/api/feedback", {
      action: "label_correction",
      event_id: state.modalEventId,
      corrected_item_id: correctedItem,
    });
    state.feedbackLog = res.feedback_log;
    state.clsIndexSummary = res.cls_index_summary;
    state.recommendationsBundle = res.recommendations;
    state.schedulePlan = await apiGet("/api/plan");
    closeLabelModal();
    renderAll();
    showToast(res.entry.detail);
  });

  // Planner controls
  $("#plannerGoalRange").addEventListener("input", (e) => {
    $("#plannerGoalVal").textContent = `${e.target.value}đ`;
  });
  $("#btnRunPlan").addEventListener("click", async () => {
    const goal = Number($("#plannerGoalRange").value);
    state.schedulePlan = await apiPost("/api/plan", { target_total: goal });
    renderPlanner();
    showToast(`Đã lập lịch không trùng giờ hướng tới ${goal}đ ĐRL`);
  });

  // Live Classifier form + quick samples
  $("#classifyForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await runLiveClassify();
  });
  document.querySelectorAll(".quick-samples .chip-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = QUICK_SAMPLES[btn.dataset.sample];
      if (!sample) return;
      $("#clsTitle").value = sample.title;
      $("#clsDesc").value = sample.description;
      $("#clsOrg").value = sample.organizer;
      $("#clsRole").value = sample.role;
      await runLiveClassify();
    });
  });

  // Profile chips & save/reset
  $("#interestChipsGrid").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip-btn");
    if (!btn) return;
    const cid = btn.dataset.chip;
    const list = new Set(state.profile.selected_chips || []);
    if (list.has(cid)) list.delete(cid);
    else list.add(cid);
    state.profile.selected_chips = Array.from(list);
    renderProfileAndFeedback();
  });

  $("#btnSaveProfile").addEventListener("click", async () => {
    const res = await apiPost("/api/profile", {
      cohort: $("#profCohort").value,
      faculty: $("#profFaculty").value,
      max_budget_vnd: Number($("#profBudget").value || 0),
      selected_chips: state.profile.selected_chips,
    });
    state.profile = res.profile;
    state.recommendationsBundle = res.recommendations;
    state.schedulePlan = await apiGet("/api/plan");
    renderAll();
    showToast("Đã cập nhật hồ sơ P4 và tính lại vector sở thích!");
  });

  $("#btnResetDemo").addEventListener("click", async () => {
    const res = await apiPost("/api/feedback", { action: "reset_demo" });
    state.feedbackLog = res.feedback_log;
    state.recommendationsBundle = res.recommendations;
    state.schedulePlan = await apiGet("/api/plan");
    renderAll();
    showToast("Đã khôi phục dữ liệu demo về trạng thái ban đầu.");
  });
}

async function init() {
  setupListeners();
  const data = await apiGet("/api/bootstrap");
  state.bootstrap = data;
  state.profile = data.profile;
  state.rubricCatalog = data.rubric_catalog;
  state.clsIndexSummary = data.cls_index_summary;
  state.feedbackLog = data.feedback_log;
  state.recommendationsBundle = data.recommendations_bundle;
  state.schedulePlan = data.schedule_plan;
  renderAll();
  await runLiveClassify();
}

init().catch((err) => {
  console.error(err);
  showToast(`Lỗi khởi tạo prototype: ${err.message}`);
});
