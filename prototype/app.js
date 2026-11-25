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
  blood: {
    title: "Ngày hội Hiến máu tình nguyện Giọt hồng HCMUS",
    description: "Chương trình hiến máu nhân đạo vì cộng đồng do Hội Sinh viên Trường tổ chức.",
    organizer: "HSV",
    role: "participant",
  },
  mobilize: {
    title: "Huy động lực lượng cổ vũ Giải Bóng đá Futsal sinh viên ĐHQG-HCM",
    description: "Thông báo huy động sinh viên tham gia cổ vũ trận đấu cấp Trường.",
    organizer: "CTSV",
    role: "participant",
  },
  quiz31_35: {
    title: "Cuộc thi trắc nghiệm trực tuyến Tìm hiểu pháp luật và an toàn không gian mạng",
    description: "Cuộc thi trực tuyến tìm hiểu quy định pháp luật và phòng chống lừa đảo trực tuyến.",
    organizer: "ĐK CNTT",
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
  dualBranchSummary: null,
  dualBranchEvents: [],
  batchStatusFilter: "ALL",
  batchSearchQuery: "",
  selectedBatchPostId: null,
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
  const db = state.dualBranchSummary;
  const prof = state.profile;
  $("#studentChip").textContent = `${prof.name} · ${prof.cohort} · ${prof.class_id}`;

  const banner = $("#embedderBanner");
  const clsBadge = $("#clsModelBadge");
  const dbInfo = db
    ? ` · Dual-Branch <strong class="mono">${escapeHtml(db.algorithm)}</strong> (Sample: ${db.sample_reference_count} · Rubric: ${db.rubric_total_prototypes} · β=${db.default_params.beta}, α=${db.default_params.alpha})`
    : "";

  if (st.is_e5_active) {
    banner.className = "status-ribbon";
    banner.innerHTML = `
      <span class="badge badge-teal mono">P2 DUAL-BRANCH ACTIVE</span>
      <span>Embedding: <strong class="mono">${escapeHtml(st.active_model)}</strong> (${st.dimension}d · L2)${dbInfo}</span>
    `;
    if (clsBadge) {
      clsBadge.className = "badge badge-teal mono";
      clsBadge.textContent = `e5-small · 2-branch · ${st.dimension}d`;
    }
  } else {
    banner.className = "status-ribbon fallback";
    banner.innerHTML = `
      <span class="badge badge-orange mono">P2.1 E5-SMALL STANDBY</span>
      <span>Mục tiêu: <strong class="mono">${escapeHtml(st.requested_model)}</strong> (384d) · Hiện chạy fallback <code class="mono">${escapeHtml(st.active_model)}</code>${dbInfo}</span>
    `;
    if (clsBadge) {
      clsBadge.className = "badge badge-orange mono";
      clsBadge.textContent = `fallback · 2-branch · ${st.dimension}d`;
    }
  }
}

function renderProgressStrip() {
  const bundle = state.recommendationsBundle;
  const led = bundle.ledger_summary;
  const db = state.dualBranchSummary;
  $("#totalScoreVal").textContent = `${led.total_score}đ`;
  $("#targetScoreVal").textContent = `${led.target_total}`;
  $("#kpiTotalEvents").textContent = bundle.stats.total_cached_events;
  $("#kpiEligible").textContent = bundle.stats.eligible_count;
  $("#kpiDropped").textContent = bundle.stats.dropped_count;
  $("#kpiClsSize").textContent = db
    ? `${db.sample_reference_count}S / ${db.rubric_total_prototypes}R`
    : `${state.clsIndexSummary.total_samples}`;
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
                ${
                  c.dual_branch_status === "PREDICTED"
                    ? `<span class="badge badge-teal mono" title="Hai nhánh đồng thuận & đạt ngưỡng tạm thời">PREDICTED</span>`
                    : c.dual_branch_status === "NEEDS_REVIEW"
                      ? `<span class="badge badge-orange mono" title="${escapeHtml((c.dual_branch_review_reasons || []).join("; "))}">NEEDS_REVIEW</span>`
                      : ""
                }
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
              <button class="btn btn-ghost" data-action="inspect_dual" data-id="${escapeHtml(c.event_id)}">
                Xem 2 nhánh
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

function statusBadgeClass(status) {
  if (status === "PREDICTED") return "badge-teal";
  if (status === "NEEDS_REVIEW") return "badge-orange";
  if (status === "NO_MATCH") return "badge-danger";
  return "badge-muted";
}

function fmtOptNum(val, digits = 4) {
  return typeof val === "number" && Number.isFinite(val) ? val.toFixed(digits) : "—";
}

function renderDualBatchBrowser() {
  const pillsEl = $("#batchStatusPills");
  const listEl = $("#batchEventsList");
  if (!pillsEl || !listEl) return;

  const counts = (state.dualBranchSummary && state.dualBranchSummary.status_counts) || {};
  const total = state.dualBranchEvents.length;
  const filters = [
    { key: "ALL", label: `Tất cả (${total})` },
    { key: "PREDICTED", label: `PREDICTED (${counts.PREDICTED || 0})` },
    { key: "NEEDS_REVIEW", label: `NEEDS_REVIEW (${counts.NEEDS_REVIEW || 0})` },
    { key: "NO_MATCH", label: `NO_MATCH (${counts.NO_MATCH || 0})` },
  ];

  pillsEl.innerHTML = filters
    .map(
      (f) => `
        <button type="button" class="chip-btn ${state.batchStatusFilter === f.key ? "selected" : ""}" data-batch-filter="${f.key}">
          ${escapeHtml(f.label)}
        </button>
      `
    )
    .join("");

  const q = state.batchSearchQuery.trim().toLowerCase();
  const filtered = state.dualBranchEvents.filter((ev) => {
    if (state.batchStatusFilter !== "ALL" && ev.status !== state.batchStatusFilter) return false;
    if (!q) return true;
    return (
      String(ev.post_id || "").toLowerCase().includes(q) ||
      String(ev.title || "").toLowerCase().includes(q) ||
      String(ev.predicted_criterion_ref || "").toLowerCase().includes(q) ||
      String(ev.top_suggested_ref || "").toLowerCase().includes(q)
    );
  });

  if (!filtered.length) {
    listEl.innerHTML = `<p class="helper-text">Không có sự kiện khớp bộ lọc.</p>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((ev) => {
      const refLabel = ev.predicted_criterion_ref
        ? `${ev.predicted_criterion_ref} (${ev.predicted_group_id})`
        : ev.top_suggested_ref
          ? `Gợi ý: ${ev.top_suggested_ref} (${ev.top_suggested_group})`
          : "—";
      const isActive = state.selectedBatchPostId === ev.post_id;
      return `
        <button type="button" class="batch-event-item ${isActive ? "active" : ""}" data-batch-post-id="${escapeHtml(ev.post_id)}">
          <div class="batch-event-head">
            <span class="mono" style="font-size:11.5px;font-weight:700">${escapeHtml(ev.post_id)} · ${escapeHtml(refLabel)}</span>
            <span class="badge ${statusBadgeClass(ev.status)} mono">${escapeHtml(ev.status)}</span>
          </div>
          <div class="batch-event-title">${escapeHtml(ev.title)}</div>
        </button>
      `;
    })
    .join("");
}

function renderClassifyTrace(res) {
  const db = res.dual_branch;
  const m6 = res.module6_credit_estimate;
  const params = res.knn_params_used || {};
  const pools = res.reference_pool_sizes || {};
  const conf = (db && db.confidence) || {};
  const scores = (db && db.scores) || {};
  const s1Fused = (scores.stage1 && scores.stage1.fused) || [];
  const s2Fused = (scores.stage2 && scores.stage2.fused) || [];
  const evi = (db && db.evidence) || {};
  const sampleNbs = evi.sample_neighbors || [];
  const rubricProtos = evi.rubric_prototypes || [];
  const rubricCtx = evi.rubric_context;
  const reasons = (db && db.review_reasons) || [];

  const status = (db && db.status) || "NEEDS_REVIEW";
  const selectedGroups = s1Fused.filter((g) => g.selected_for_stage2).map((g) => g.group_id);
  const topCandidate = s2Fused[0] || null;

  const decisionHeadline =
    status === "PREDICTED"
      ? `<strong class="mono text-teal">${escapeHtml(db.predicted_criterion_ref)} (${escapeHtml(db.predicted_group_id)})</strong> — ${escapeHtml(db.predicted_criterion_name || "")}`
      : status === "NEEDS_REVIEW"
        ? `Chưa tự động chốt nhãn (<code class="mono">predicted_criterion_ref = null</code>). Ứng viên #1: <strong class="mono">${escapeHtml(topCandidate ? `${topCandidate.criterion_ref} (${topCandidate.group_id})` : "—")}</strong>`
        : status === "NO_MATCH"
          ? `Không tìm thấy tiêu chí đủ hỗ trợ (<code class="mono">NO_MATCH</code>; không tự gán <code class="mono">NONE</code>).`
          : `Bỏ qua (không phải sự kiện hợp lệ).`;

  const agreementBadge =
    conf.branch_agreement === true
      ? `<span class="badge badge-teal">2 nhánh đồng thuận</span>`
      : conf.branch_agreement === false
        ? `<span class="badge badge-orange">2 nhánh bất đồng</span>`
        : `<span class="badge badge-muted">Thiếu 1 nhánh tham chiếu</span>`;

  $("#classifyResultPane").innerHTML = `
    <div class="decision-banner status-${escapeHtml(status)}">
      <div class="card-title-row">
        <div class="rec-badges">
          <span class="badge ${statusBadgeClass(status)} mono" style="font-size:12.5px">${escapeHtml(status)}</span>
          <span class="badge badge-muted mono">${escapeHtml((db && db.algorithm) || "hierarchical_two_stage_knn")}</span>
          ${agreementBadge}
          <span class="badge badge-muted mono">calibrated: false</span>
        </div>
        <span class="mono tag-soft">margin G=${fmtOptNum(conf.group_margin, 3)} · C=${fmtOptNum(conf.criterion_margin, 3)}</span>
      </div>
      <div style="font-size:14px;margin-top:4px">${decisionHeadline}</div>
      ${
        reasons.length
          ? `<ul class="review-reasons-list">${reasons.map((r) => `<li><code class="mono">${escapeHtml(r)}</code></li>`).join("")}</ul>`
          : ""
      }
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 1 · Shared Query Embedding (${escapeHtml(res.embedder_status.active_model)})</h2>
        <span class="mono tag-soft">${res.vector_dimension}d · ||v||₂ = ${res.vector_l2_norm} · ${res.elapsed_ms}ms${res.reused_cached_query ? " · cached" : ""}</span>
      </div>
      <p class="helper-text">Đúng 1 vector truy vấn duy nhất (chỉ dùng tiêu đề + mô tả, không rò rỉ nhãn) được chia sẻ cho cả 2 nhánh Sample &amp; Rubric ở cả 2 tầng.</p>
      <div class="query-preview mono">${escapeHtml(res.formatted_query_text)}</div>
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 2 · Stage 1 — Dual-Branch Major Groups + Fusion (β = ${params.beta ?? 0.5})</h2>
        <span class="badge badge-teal mono">Shared Top-${params.stage1_top_k ?? 2}: ${escapeHtml(selectedGroups.join(", "))}</span>
      </div>
      <p class="helper-text">
        Sample Branch (Top-K1=${params.k1 ?? 10} trên ${pools.sample_stage1_total ?? 127} mẫu) và Rubric Branch (${pools.rubric_stage1_total ?? 4} prototype nhóm <code class="mono">stage=group</code>) chạy độc lập, chuẩn hóa rồi hợp nhất: <code class="mono">S1(g) = β·S_sample + (1−β)·S_rubric</code>.
      </p>
      <table class="dual-table">
        <thead>
          <tr>
            <th>Nhóm</th>
            <th>Sample raw / norm</th>
            <th>Rubric raw / norm</th>
            <th>Fused S1(g)</th>
            <th>Stage 2</th>
          </tr>
        </thead>
        <tbody>
          ${s1Fused
            .map(
              (g) => `
                <tr class="${g.selected_for_stage2 ? "selected-row" : ""}">
                  <td><strong class="mono">${escapeHtml(g.group_id)}</strong> · ${escapeHtml(GROUP_NAMES[g.group_id] || "")}</td>
                  <td class="mono">${fmtOptNum(g.sample_raw_score, 3)} / ${fmtOptNum(g.sample_norm_score, 3)}</td>
                  <td class="mono">${fmtOptNum(g.rubric_raw_score, 3)} / ${fmtOptNum(g.rubric_norm_score, 3)}</td>
                  <td class="mono"><strong>${fmtOptNum(g.fused_score, 4)}</strong></td>
                  <td>${g.selected_for_stage2 ? '<span class="badge badge-teal">Top-2 chung</span>' : '<span class="text-muted">—</span>'}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>

    <div class="trace-step highlight">
      <div class="card-title-row">
        <h2 class="card-title">Bước 3 · Stage 2 — Dual-Branch Criteria trong {${escapeHtml(selectedGroups.join(", "))}} + Fusion (α = ${params.alpha ?? 0.5})</h2>
        <span class="mono tag-soft">Sample pool: ${pools.sample_stage2_filtered ?? "?"} · Rubric pool: ${pools.rubric_stage2_filtered ?? "?"}</span>
      </div>
      <p class="helper-text">
        Cả 2 nhánh đều lọc đúng trên <strong>Top-2 nhóm chung</strong>. Căn khớp ứng viên theo <code class="mono">criterion_ref</code>; tiêu chí chưa có mẫu trong <code class="mono">events_full.csv</code> được đánh dấu <code class="mono">no_reference_samples</code> (không coi là vô nghĩa).
      </p>
      <table class="dual-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Tiêu chí</th>
            <th>Sample (K2=${params.k2 ?? 10})</th>
            <th>Rubric</th>
            <th>Fused S2(c)</th>
            <th>Bằng chứng</th>
          </tr>
        </thead>
        <tbody>
          ${s2Fused
            .slice(0, 8)
            .map((c) => {
              const sampleCell =
                c.sample_evidence_status === "no_reference_samples"
                  ? `<span class="badge badge-muted mono">no_samples</span>`
                  : c.sample_evidence_status === "not_in_top_k"
                    ? `<span class="text-muted mono">not_in_topK</span>`
                    : `<span class="mono">${fmtOptNum(c.sample_raw_score, 3)}</span>`;
              const verBadge =
                c.evidence_verification_status === "verified_from_school_list"
                  ? `<span class="badge badge-teal mono">verified</span>`
                  : c.evidence_verification_status === "verified_partial"
                    ? `<span class="badge badge-orange mono">partial</span>`
                    : `<span class="badge badge-orange mono">draft_only</span>`;
              return `
                <tr class="${c.rank === 1 ? "selected-row" : ""}">
                  <td class="mono">#${c.rank}</td>
                  <td>
                    <strong class="mono">${escapeHtml(c.criterion_ref)}</strong>
                    <span class="text-muted mono">(${escapeHtml(c.group_id)})</span>
                    <div style="font-size:11.5px;color:#475569">${escapeHtml((c.criterion_name || "").slice(0, 72))}</div>
                  </td>
                  <td>${sampleCell}</td>
                  <td class="mono">${fmtOptNum(c.rubric_raw_score, 3)}</td>
                  <td class="mono"><strong>${fmtOptNum(c.fused_score, 4)}</strong></td>
                  <td>${verBadge}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 4 · Bằng chứng Truy xuất Độc lập (Sample &amp; Rubric Evidence)</h2>
        <span class="mono tag-soft">Cosine dot product</span>
      </div>
      <div class="evidence-grid">
        <div class="evidence-col">
          <div class="card-subtitle">Top Sample Neighbors (${sampleNbs.length})</div>
          ${
            sampleNbs.length
              ? sampleNbs
                  .map(
                    (nb) => `
                      <div class="evidence-item">
                        <div class="card-title-row" style="margin-bottom:2px">
                          <span class="mono"><strong>${escapeHtml(nb.criterion_ref)}</strong> · ${escapeHtml(nb.sample_id)}</span>
                          <span class="mono text-teal"><strong>${fmtOptNum(nb.similarity, 4)}</strong></span>
                        </div>
                        <div>${escapeHtml(nb.event_title)}</div>
                      </div>
                    `
                  )
                  .join("")
              : `<p class="helper-text" style="margin:0">Không có mẫu tương đồng trong Top-2 nhóm.</p>`
          }
        </div>
        <div class="evidence-col">
          <div class="card-subtitle">Top Rubric Prototypes (${rubricProtos.length})</div>
          ${
            rubricProtos
              .length
              ? rubricProtos
                  .map(
                    (p) => `
                      <div class="evidence-item">
                        <div class="card-title-row" style="margin-bottom:2px">
                          <span class="mono"><strong>${escapeHtml(p.criterion_ref)}</strong> · ${escapeHtml(p.prototype_type)}</span>
                          <span class="mono text-teal"><strong>${fmtOptNum(p.similarity, 4)}</strong></span>
                        </div>
                        <div class="text-muted mono" style="font-size:11px">${escapeHtml(p.prototype_id)} · status=${escapeHtml(p.anchor_status || "-")}</div>
                      </div>
                    `
                  )
                  .join("")
              : `<p class="helper-text" style="margin:0">Không có prototype phù hợp.</p>`
          }
        </div>
      </div>
      ${
        rubricCtx
          ? `
            <div class="rubric-context-box">
              <div><strong>Ngữ cảnh Rubric (${escapeHtml(rubricCtx.criterion_ref)} · ${escapeHtml(rubricCtx.criterion_key)}):</strong> <span class="badge badge-muted mono">anchors_status: ${escapeHtml(rubricCtx.anchors_status || "-")}</span></div>
              ${rubricCtx.scope_note ? `<div style="margin-top:4px">${escapeHtml(rubricCtx.scope_note)}</div>` : ""}
              ${
                rubricCtx.open_questions && rubricCtx.open_questions.length
                  ? `<div style="margin-top:6px;font-size:12px;color:#9A3412"><strong>Open questions:</strong> ${rubricCtx.open_questions.map((q) => `${escapeHtml(q.id)} (${escapeHtml(q.topic)})`).join(" · ")}</div>`
                  : ""
              }
            </div>
          `
          : ""
      }
    </div>

    <div class="trace-step">
      <div class="card-title-row">
        <h2 class="card-title">Bước 5 · MODULE 6 — Ước tính điểm quy chế (Tách biệt theo D-3)</h2>
        <span class="badge ${m6.is_credit_eligible ? "badge-teal" : "badge-muted"}">${m6.is_credit_eligible ? "Đủ điều kiện cộng điểm" : "Không cộng điểm / Đã chạm trần"}</span>
      </div>
      <div class="kpi-strip">
        <div class="kpi-box">
          <div class="kpi-label">Mã mục tham chiếu</div>
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
  renderDualBatchBrowser();
  renderProfileAndFeedback();
}

function switchTab(target) {
  document.querySelectorAll(".nav-tab").forEach((b) => {
    const active = b.dataset.tab === target;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${target}`);
  });
}

async function inspectEventByPostId(postId) {
  const ev = state.dualBranchEvents.find((item) => item.post_id === postId);
  if (!ev) return;
  state.selectedBatchPostId = postId;
  $("#clsPostId").value = ev.post_id || "";
  $("#clsTitle").value = ev.title || "";
  $("#clsDesc").value = ev.description || "";
  $("#clsOrg").value = ev.organizer || "";
  $("#clsRole").value = "participant";
  renderDualBatchBrowser();
  await runLiveClassify({ useCachedQuery: true });
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

async function runLiveClassify(options = {}) {
  const payload = {
    post_id: $("#clsPostId") ? $("#clsPostId").value : "",
    use_cached_query: Boolean(options.useCachedQuery),
    title: $("#clsTitle").value,
    description: $("#clsDesc").value,
    organizer: $("#clsOrg").value,
    role: $("#clsRole").value,
    stage1_top_k: Number($("#paramTopK").value || 2),
    k1: Number($("#paramK1").value || 10),
    k2: Number($("#paramK2").value || 10),
    beta: Number($("#paramBeta") ? $("#paramBeta").value : 0.5),
    alpha: Number($("#paramAlpha") ? $("#paramAlpha").value : 0.5),
  };
  const res = await apiPost("/api/classify", payload);
  renderClassifyTrace(res);
}

function setupListeners() {
  // Navigation tabs
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
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
    if (action === "inspect_dual") {
      switchTab("classifier");
      await inspectEventByPostId(eventId);
      return;
    }
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

  // Live Classifier form + quick samples + 108-event batch browser
  $("#classifyForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    if ($("#clsPostId")) $("#clsPostId").value = "";
    state.selectedBatchPostId = null;
    renderDualBatchBrowser();
    await runLiveClassify({ useCachedQuery: false });
  });
  document.querySelectorAll(".quick-samples .chip-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = QUICK_SAMPLES[btn.dataset.sample];
      if (!sample) return;
      if ($("#clsPostId")) $("#clsPostId").value = "";
      state.selectedBatchPostId = null;
      $("#clsTitle").value = sample.title;
      $("#clsDesc").value = sample.description;
      $("#clsOrg").value = sample.organizer;
      $("#clsRole").value = sample.role;
      renderDualBatchBrowser();
      await runLiveClassify({ useCachedQuery: false });
    });
  });

  if ($("#batchStatusPills")) {
    $("#batchStatusPills").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-batch-filter]");
      if (!btn) return;
      state.batchStatusFilter = btn.dataset.batchFilter;
      renderDualBatchBrowser();
    });
  }
  if ($("#batchSearchInput")) {
    $("#batchSearchInput").addEventListener("input", (e) => {
      state.batchSearchQuery = e.target.value;
      renderDualBatchBrowser();
    });
  }
  if ($("#batchEventsList")) {
    $("#batchEventsList").addEventListener("click", async (e) => {
      const item = e.target.closest("button[data-batch-post-id]");
      if (!item) return;
      await inspectEventByPostId(item.dataset.batchPostId);
    });
  }
  if ($("#btnRerunPipeline")) {
    $("#btnRerunPipeline").addEventListener("click", async () => {
      const btn = $("#btnRerunPipeline");
      btn.disabled = true;
      btn.textContent = "Đang chạy...";
      try {
        const res = await apiPost("/api/dual-branch/run", {});
        state.dualBranchSummary = res.dual_branch_events.summary;
        state.dualBranchEvents = res.dual_branch_events.events;
        state.recommendationsBundle = res.recommendations_bundle;
        state.schedulePlan = res.schedule_plan;
        renderAll();
        showToast(
          `Đã chạy lại pipeline (${res.algorithm}): ${res.status_counts.PREDICTED} PREDICTED, ${res.status_counts.NEEDS_REVIEW} NEEDS_REVIEW`
        );
      } finally {
        btn.disabled = false;
        btn.textContent = "Chạy lại Pipeline";
      }
    });
  }

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
  state.dualBranchSummary = data.dual_branch_summary || null;
  state.dualBranchEvents = (data.dual_branch_events && data.dual_branch_events.events) || [];
  state.feedbackLog = data.feedback_log;
  state.recommendationsBundle = data.recommendations_bundle;
  state.schedulePlan = data.schedule_plan;

  // Sync default provisional params from classification_config.yaml
  if (state.dualBranchSummary && state.dualBranchSummary.default_params) {
    const dp = state.dualBranchSummary.default_params;
    if ($("#paramTopK")) $("#paramTopK").value = dp.stage1_top_k;
    if ($("#paramK1")) $("#paramK1").value = dp.k1;
    if ($("#paramK2")) $("#paramK2").value = dp.k2;
    if ($("#paramBeta")) $("#paramBeta").value = dp.beta;
    if ($("#paramAlpha")) $("#paramAlpha").value = dp.alpha;
  }

  renderAll();
  if (state.dualBranchEvents.length) {
    await inspectEventByPostId(state.dualBranchEvents[0].post_id);
  } else {
    await runLiveClassify();
  }
}

init().catch((err) => {
  console.error(err);
  showToast(`Lỗi khởi tạo prototype: ${err.message}`);
});
