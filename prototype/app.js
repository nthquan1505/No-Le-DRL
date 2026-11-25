/* No-Le-DRL Frontend Controller — Dual-Branch Classification + Module 6 */

const $ = (sel) => document.querySelector(sel);

const QUICK_SAMPLES = {
  ctf: {
    title: "Cuộc thi An toàn thông tin HCMUS CTF 2026",
    description: "Cuộc thi học thuật cấp Khoa dành cho sinh viên yêu thích bảo mật và lập trình.",
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
    title: "Chiến dịch Tình nguyện Mùa Hè Xanh 2026 (3 ngày)",
    description: "Hoạt động tình nguyện vì cộng đồng kéo dài 3 ngày tại xã vùng ven.",
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
};

const GROUP_NAMES = {
  C1: "Ý thức & học tập",
  C2: "Chấp hành nội quy",
  C3: "Chính trị – Xã hội – Văn thể",
  C4: "Phẩm chất công dân",
  C5: "Khuyến khích",
  P: "Điểm trừ / Nghĩa vụ",
};

const state = {
  bootstrap: null,
  dualBranchSummary: null,
  dualBranchEvents: [],
  batchStatusFilter: "ALL",
  batchSearchQuery: "",
  selectedBatchPostId: null,
  selectedBatchEventDetail: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtOptNum(val, digits = 3) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
  return Number(val).toFixed(digits);
}

function statusBadgeClass(status) {
  if (status === "PREDICTED") return "badge-teal";
  if (status === "NEEDS_REVIEW") return "badge-orange";
  if (status === "NO_MATCH" || status === "SKIPPED_NOT_EVENT") return "badge-danger";
  return "badge-muted";
}

function showToast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    el.hidden = true;
  }, 2600);
}

async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
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
   SHARED RESULT RENDERER (Tab 1 & Tab 2)
   ========================================================================= */

function renderClassificationResult(res) {
  const db = res.dual_branch || {};
  const m6 = res.module6_credit_estimate || {};
  const m6Mode = res.module6_ref_mode || "none";
  const params = res.knn_params_used || {};
  const pools = res.reference_pool_sizes || {};
  const conf = db.confidence || {};
  const scores = db.scores || {};
  const s1Fused = (scores.stage1 && scores.stage1.fused) || [];
  const s2Fused = (scores.stage2 && scores.stage2.fused) || [];
  const evi = db.evidence || {};
  const sampleNbs = evi.sample_neighbors || [];
  const rubricProtos = evi.rubric_prototypes || [];
  const rubricCtx = evi.rubric_context;
  const reasons = db.review_reasons || [];

  const status = db.status || "NEEDS_REVIEW";
  const selectedGroups = s1Fused.filter((g) => g.selected_for_stage2).map((g) => g.group_id);
  const topCandidate = s2Fused[0] || null;

  let headlineHtml = "";
  let sublineHtml = "";
  if (status === "PREDICTED") {
    headlineHtml = `<span class="mono text-teal">${escapeHtml(db.predicted_criterion_ref)} (${escapeHtml(db.predicted_group_id)})</span> — ${escapeHtml(db.predicted_criterion_name || "")}`;
    sublineHtml = "Hai nhánh đồng thuận và đạt ngưỡng kiểm tra tạm thời.";
  } else if (status === "NEEDS_REVIEW") {
    headlineHtml = "Cần kiểm tra thủ công";
    sublineHtml = topCandidate
      ? `Ứng viên #1: <strong class="mono">${escapeHtml(topCandidate.criterion_ref)} (${escapeHtml(topCandidate.group_id)})</strong> — ${escapeHtml(topCandidate.criterion_name || "")}`
      : "Chưa đủ căn cứ để tự động chốt tiêu chí.";
  } else if (status === "NO_MATCH") {
    headlineHtml = "Không khớp tiêu chí ĐRL phù hợp";
    sublineHtml = "Độ tương đồng thấp hơn ngưỡng tối thiểu (không tự gán nhãn NONE).";
  } else {
    headlineHtml = "Bỏ qua bản ghi";
    sublineHtml = "Không phải thông báo sự kiện hợp lệ hoặc thiếu nội dung.";
  }

  const agreementBadge =
    conf.branch_agreement === true
      ? `<span class="badge badge-teal">2 nhánh đồng thuận</span>`
      : conf.branch_agreement === false
        ? `<span class="badge badge-orange">2 nhánh bất đồng</span>`
        : `<span class="badge badge-muted">Thiếu 1 nhánh tham chiếu</span>`;

  const pointsText =
    m6.min_points !== undefined &&
    m6.max_points !== undefined &&
    Number(m6.min_points) !== Number(m6.max_points)
      ? `+${m6.expected_points}đ (${m6.min_points}–${m6.max_points}đ)`
      : `+${m6.expected_points ?? 0}đ`;

  const m6ModeBadge =
    m6Mode === "predicted"
      ? `<span class="badge badge-teal">Theo nhãn dự đoán</span>`
      : m6Mode === "suggested_preview"
        ? `<span class="badge badge-orange">Xem trước theo ứng viên #1</span>`
        : `<span class="badge badge-muted">Chưa xác định mục</span>`;

  return `
    <div class="card decision-card status-${escapeHtml(status)}">
      <div class="card-title-row">
        <div class="badge-row">
          <span class="badge ${statusBadgeClass(status)} mono">${escapeHtml(status)}</span>
          ${agreementBadge}
          <span class="badge badge-muted mono">provisional · chưa hiệu chỉnh</span>
        </div>
        <span class="mono text-muted" style="font-size:12px">${escapeHtml(res.elapsed_ms)} ms</span>
      </div>
      <div class="decision-headline">${headlineHtml}</div>
      <div class="decision-subline">${sublineHtml}</div>
      ${
        reasons.length
          ? `<ul class="review-reasons-list">${reasons.map((r) => `<li><code class="mono">${escapeHtml(r)}</code></li>`).join("")}</ul>`
          : ""
      }
    </div>

    <div class="card">
      <div class="card-title-row">
        <h2 class="card-title">Ước tính điểm ĐRL (Module 6)</h2>
        <div class="badge-row">
          ${m6ModeBadge}
          <span class="badge ${m6.is_credit_eligible ? "badge-teal" : "badge-muted"}">
            ${m6.is_credit_eligible ? "Đủ điều kiện cộng điểm" : "Không cộng điểm"}
          </span>
        </div>
      </div>
      <div class="kpi-strip">
        <div class="kpi-box">
          <div class="kpi-label">Mã mục áp dụng</div>
          <div class="kpi-val mono">${escapeHtml(m6.rubric_item_id || "—")} ${m6.group_id ? `(${escapeHtml(m6.group_id)})` : ""}</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Điểm quy chế dự kiến</div>
          <div class="kpi-val mono text-teal">${escapeHtml(pointsText)}</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Nguồn tính</div>
          <div class="kpi-val mono">${escapeHtml(m6.source || "—")}</div>
        </div>
      </div>
      ${res.rubric_item_name ? `<p class="helper-text" style="margin-top:10px">${escapeHtml(res.rubric_item_name)}</p>` : ""}
      ${
        m6.warning_flags && m6.warning_flags.length
          ? `<div class="badge-row" style="margin-top:10px">${m6.warning_flags.map((f) => `<span class="badge badge-muted mono">${escapeHtml(f)}</span>`).join("")}</div>`
          : ""
      }
    </div>

    <details class="tech-details">
      <summary class="tech-details-summary">
        <span>Chi tiết kỹ thuật 2 nhánh &amp; Bằng chứng</span>
        <span class="mono text-muted" style="font-size:11.5px">
          margin G=${fmtOptNum(conf.group_margin, 3)} · C=${fmtOptNum(conf.criterion_margin, 3)} · Top-${params.stage1_top_k ?? 2}: ${escapeHtml(selectedGroups.join(", ") || "—")}
        </span>
      </summary>
      <div class="tech-details-body">
        <div class="tech-section">
          <div class="card-title-row">
            <h3 class="card-subtitle">Shared Query Vector (${escapeHtml(res.embedder_status?.active_model || "e5-small")})</h3>
            <span class="mono text-muted" style="font-size:11.5px">${res.vector_dimension}d · ||v||₂ = ${res.vector_l2_norm} · ${res.reused_cached_query ? "cached" : "live"}</span>
          </div>
          <div class="query-preview mono">${escapeHtml(res.formatted_query_text)}</div>
        </div>

        <div class="tech-section">
          <div class="card-title-row">
            <h3 class="card-subtitle">Stage 1 · Nhóm lớn (K1=${params.k1 ?? 10}, β=${params.beta ?? 0.5})</h3>
            <span class="badge badge-teal mono">Top-${params.stage1_top_k ?? 2}: ${escapeHtml(selectedGroups.join(", "))}</span>
          </div>
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
                      <td>${g.selected_for_stage2 ? '<span class="badge badge-teal">Chọn</span>' : '<span class="text-muted">—</span>'}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>

        <div class="tech-section">
          <div class="card-title-row">
            <h3 class="card-subtitle">Stage 2 · Tiêu chí (K2=${params.k2 ?? 10}, α=${params.alpha ?? 0.5})</h3>
            <span class="mono text-muted" style="font-size:11.5px">Sample pool: ${pools.sample_stage2_filtered ?? "?"} · Rubric pool: ${pools.rubric_stage2_filtered ?? "?"}</span>
          </div>
          <table class="dual-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Tiêu chí</th>
                <th>Sample</th>
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
                        <div class="text-muted" style="font-size:11.5px">${escapeHtml((c.criterion_name || "").slice(0, 72))}</div>
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

        <div class="tech-section">
          <h3 class="card-subtitle">Bằng chứng truy xuất độc lập</h3>
          <div class="evidence-grid">
            <div class="evidence-col">
              <div class="card-subtitle">Top Sample Neighbors (${sampleNbs.length})</div>
              ${
                sampleNbs.length
                  ? sampleNbs
                      .map(
                        (nb) => `
                          <div class="evidence-item">
                            <div class="card-title-row">
                              <span class="mono"><strong>${escapeHtml(nb.criterion_ref)}</strong> · ${escapeHtml(nb.sample_id)}</span>
                              <span class="mono text-teal"><strong>${fmtOptNum(nb.similarity, 4)}</strong></span>
                            </div>
                            <div>${escapeHtml(nb.event_title)}</div>
                          </div>
                        `
                      )
                      .join("")
                  : `<p class="helper-text">Không có mẫu tương đồng trong Top-k nhóm.</p>`
              }
            </div>
            <div class="evidence-col">
              <div class="card-subtitle">Top Rubric Prototypes (${rubricProtos.length})</div>
              ${
                rubricProtos.length
                  ? rubricProtos
                      .map(
                        (p) => `
                          <div class="evidence-item">
                            <div class="card-title-row">
                              <span class="mono"><strong>${escapeHtml(p.criterion_ref)}</strong> · ${escapeHtml(p.prototype_type)}</span>
                              <span class="mono text-teal"><strong>${fmtOptNum(p.similarity, 4)}</strong></span>
                            </div>
                            <div class="text-muted mono" style="font-size:11px">${escapeHtml(p.prototype_id)} · ${escapeHtml(p.anchor_status || "-")}</div>
                          </div>
                        `
                      )
                      .join("")
                  : `<p class="helper-text">Không có prototype phù hợp.</p>`
              }
            </div>
          </div>
          ${
            rubricCtx
              ? `
                <div class="rubric-context-box">
                  <div><strong>Ngữ cảnh Rubric (${escapeHtml(rubricCtx.criterion_ref)} · ${escapeHtml(rubricCtx.criterion_key)}):</strong> <span class="badge badge-muted mono">${escapeHtml(rubricCtx.anchors_status || "-")}</span></div>
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
      </div>
    </details>
  `;
}

/* =========================================================================
   TAB SWITCHING & TAB 1 FORM
   ========================================================================= */

function switchTab(target) {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    const active = btn.dataset.tab === target;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${target}`);
  });
}

function syncDefaultParams() {
  const dp = state.dualBranchSummary?.default_params;
  if (!dp) return;
  $("#paramTopK").value = dp.stage1_top_k;
  $("#paramK1").value = dp.k1;
  $("#paramK2").value = dp.k2;
  $("#paramBeta").value = dp.beta;
  $("#paramAlpha").value = dp.alpha;
}

async function runLiveClassify(options = {}) {
  const btn = $("#btnClassifySubmit");
  btn.disabled = true;
  try {
    const payload = {
      post_id: $("#clsPostId").value,
      use_cached_query: Boolean(options.useCachedQuery),
      title: $("#clsTitle").value,
      description: $("#clsDesc").value,
      organizer: $("#clsOrg").value,
      role: $("#clsRole").value,
      stage1_top_k: Number($("#paramTopK").value || 2),
      k1: Number($("#paramK1").value || 10),
      k2: Number($("#paramK2").value || 10),
      beta: Number($("#paramBeta").value ?? 0.5),
      alpha: Number($("#paramAlpha").value ?? 0.5),
    };
    const res = await apiPost("/api/classify", payload);
    $("#classifyResultPane").innerHTML = renderClassificationResult(res);
  } catch (err) {
    showToast(`Lỗi phân loại: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

/* =========================================================================
   TAB 2: 108-EVENT BATCH BROWSER
   ========================================================================= */

function renderBatchBrowser() {
  const summary = state.dualBranchSummary;
  const counts = (summary && summary.status_counts) || {};
  const totalCount = state.dualBranchEvents.length;
  const skippedOrNoMatch = (counts.NO_MATCH || 0) + (counts.SKIPPED_NOT_EVENT || 0);

  const pills = [
    { key: "ALL", label: `Tất cả (${totalCount})` },
    { key: "PREDICTED", label: `PREDICTED (${counts.PREDICTED || 0})` },
    { key: "NEEDS_REVIEW", label: `NEEDS_REVIEW (${counts.NEEDS_REVIEW || 0})` },
    { key: "OTHER", label: `NO_MATCH / Bỏ qua (${skippedOrNoMatch})` },
  ];

  $("#batchStatusPills").innerHTML = pills
    .map(
      (p) => `
        <button type="button" class="filter-pill ${state.batchStatusFilter === p.key ? "active" : ""}" data-batch-filter="${p.key}">
          ${escapeHtml(p.label)}
        </button>
      `
    )
    .join("");

  const q = state.batchSearchQuery.trim().toLowerCase();
  const filtered = state.dualBranchEvents.filter((ev) => {
    if (state.batchStatusFilter === "OTHER") {
      if (ev.status !== "NO_MATCH" && ev.status !== "SKIPPED_NOT_EVENT") return false;
    } else if (state.batchStatusFilter !== "ALL" && ev.status !== state.batchStatusFilter) {
      return false;
    }
    if (!q) return true;
    const ref = ev.predicted_criterion_ref || ev.top_suggested_ref || "";
    const hay = `${ev.post_id} ${ev.title} ${ev.organizer} ${ref}`.toLowerCase();
    return hay.includes(q);
  });

  $("#batchCountLabel").textContent = `${filtered.length} / ${totalCount} sự kiện`;

  if (!filtered.length) {
    $("#batchEventsList").innerHTML = `<div class="empty-state">Không có sự kiện khớp bộ lọc.</div>`;
    return;
  }

  $("#batchEventsList").innerHTML = filtered
    .map((ev) => {
      const refLabel = ev.predicted_criterion_ref
        ? `${ev.predicted_criterion_ref} (${ev.predicted_group_id})`
        : ev.top_suggested_ref
          ? `Gợi ý #1: ${ev.top_suggested_ref} (${ev.top_suggested_group})`
          : "—";
      const active = state.selectedBatchPostId === ev.post_id ? "active" : "";
      return `
        <button type="button" class="batch-event-item ${active}" data-batch-post-id="${escapeHtml(ev.post_id)}">
          <div class="batch-event-head">
            <span class="mono text-muted" style="font-size:11.5px">${escapeHtml(ev.post_id)}</span>
            <span class="badge ${statusBadgeClass(ev.status)} mono">${escapeHtml(ev.status)}</span>
          </div>
          <div class="batch-event-title">${escapeHtml(ev.title)}</div>
          <div class="batch-event-meta">
            <span class="mono">${escapeHtml(refLabel)}</span>
            <span>${escapeHtml(ev.organizer || "")}</span>
          </div>
        </button>
      `;
    })
    .join("");
}

async function selectBatchEvent(postId) {
  const ev = state.dualBranchEvents.find((item) => item.post_id === postId);
  if (!ev) return;
  state.selectedBatchPostId = postId;
  renderBatchBrowser();

  $("#batchDetailPane").innerHTML = `
    <div class="card event-meta-card">
      <div class="card-title-row">
        <span class="mono text-muted" style="font-size:12px">${escapeHtml(ev.post_id)}${ev.date ? ` · ${escapeHtml(ev.date)}` : ""}</span>
        <button id="btnOpenInClassify" type="button" class="btn btn-ghost btn-sm">Mở sang form thử nghiệm</button>
      </div>
      <h2 class="card-title" style="margin-top:8px">${escapeHtml(ev.title)}</h2>
      ${ev.organizer ? `<p class="helper-text">Đơn vị tổ chức: <strong>${escapeHtml(ev.organizer)}</strong></p>` : ""}
      ${ev.description ? `<p class="event-desc">${escapeHtml(ev.description)}</p>` : ""}
    </div>
    <div id="batchDetailResult" class="result-col">
      <div class="card empty-state">Đang tải chi tiết và ước tính điểm Module 6...</div>
    </div>
  `;

  try {
    const res = await apiPost("/api/classify", {
      post_id: ev.post_id || "",
      use_cached_query: true,
      title: ev.title || "",
      description: ev.description || "",
      organizer: ev.organizer || "",
      role: "participant",
    });
    state.selectedBatchEventDetail = { event: ev, result: res };
    if (state.selectedBatchPostId === postId) {
      $("#batchDetailResult").innerHTML = renderClassificationResult(res);
    }
  } catch (err) {
    $("#batchDetailResult").innerHTML = `<div class="card empty-state">Không tải được chi tiết: ${escapeHtml(err.message)}</div>`;
  }
}

/* =========================================================================
   EVENT LISTENERS & INIT
   ========================================================================= */

function setupListeners() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });

  // Clear cached post_id if user edits title or description in Tab 1
  ["#clsTitle", "#clsDesc"].forEach((sel) => {
    $(sel).addEventListener("input", () => {
      $("#clsPostId").value = "";
    });
  });

  $("#classifyForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const useCache = Boolean($("#clsPostId").value);
    await runLiveClassify({ useCachedQuery: useCache });
  });

  document.querySelectorAll(".quick-samples .chip-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = QUICK_SAMPLES[btn.dataset.sample];
      if (!sample) return;
      $("#clsPostId").value = "";
      $("#clsTitle").value = sample.title;
      $("#clsDesc").value = sample.description;
      $("#clsOrg").value = sample.organizer;
      $("#clsRole").value = sample.role;
      await runLiveClassify({ useCachedQuery: false });
    });
  });

  $("#btnResetParams").addEventListener("click", () => {
    syncDefaultParams();
    showToast("Đã đặt lại tham số mặc định từ classification_config.yaml");
  });

  $("#batchStatusPills").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-batch-filter]");
    if (!btn) return;
    state.batchStatusFilter = btn.dataset.batchFilter;
    renderBatchBrowser();
  });

  $("#batchSearchInput").addEventListener("input", (e) => {
    state.batchSearchQuery = e.target.value;
    renderBatchBrowser();
  });

  $("#batchEventsList").addEventListener("click", async (e) => {
    const item = e.target.closest("button[data-batch-post-id]");
    if (!item) return;
    await selectBatchEvent(item.dataset.batchPostId);
  });

  $("#batchDetailPane").addEventListener("click", async (e) => {
    const btn = e.target.closest("#btnOpenInClassify");
    if (!btn || !state.selectedBatchEventDetail) return;
    const { event: ev, result } = state.selectedBatchEventDetail;
    $("#clsPostId").value = ev.post_id || "";
    $("#clsTitle").value = ev.title || "";
    $("#clsDesc").value = ev.description || "";
    $("#clsOrg").value = ev.organizer || "";
    $("#clsRole").value = "participant";
    syncDefaultParams();
    switchTab("classify");
    $("#classifyResultPane").innerHTML = renderClassificationResult(result);
    showToast(`Đã chuyển ${ev.post_id} sang form thử nghiệm`);
  });

  $("#btnRerunPipeline").addEventListener("click", async () => {
    const btn = $("#btnRerunPipeline");
    btn.disabled = true;
    btn.textContent = "Đang chạy...";
    try {
      const res = await apiPost("/api/dual-branch/run", {});
      state.dualBranchSummary = res.dual_branch_events.summary;
      state.dualBranchEvents = res.dual_branch_events.events;
      renderBatchBrowser();
      if (state.selectedBatchPostId) {
        await selectBatchEvent(state.selectedBatchPostId);
      }
      showToast(
        `Pipeline hoàn tất: ${res.status_counts.PREDICTED} PREDICTED, ${res.status_counts.NEEDS_REVIEW} NEEDS_REVIEW`
      );
    } catch (err) {
      showToast(`Lỗi chạy pipeline: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Chạy lại Pipeline";
    }
  });
}

async function init() {
  setupListeners();
  const data = await apiGet("/api/bootstrap");
  state.bootstrap = data;
  state.dualBranchSummary = data.dual_branch_summary || null;
  state.dualBranchEvents = (data.dual_branch_events && data.dual_branch_events.events) || [];

  const st = data.embedder_status;
  if (st) {
    $("#modelStatusPill").textContent = `${st.active_model} · ${st.dimension}d`;
  }

  syncDefaultParams();
  renderBatchBrowser();

  await runLiveClassify({ useCachedQuery: false });
  if (state.dualBranchEvents.length) {
    await selectBatchEvent(state.dualBranchEvents[0].post_id);
  }
}

init().catch((err) => {
  console.error(err);
  $("#classifyResultPane").innerHTML = `<div class="card empty-state">Không kết nối được backend: ${escapeHtml(err.message)}</div>`;
});
