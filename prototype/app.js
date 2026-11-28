/* No-Le-DRL Frontend Controller — Dual-Branch Classification + Module 6 */

const $ = (sel) => document.querySelector(sel);

const QUICK_SAMPLES = {
  ctf: {
    title: "Cuộc thi An toàn thông tin HCMUS CTF 2026",
    description: "Cuộc thi học thuật cấp Khoa dành cho sinh viên yêu thích bảo mật và lập trình.",
    format: "contest",
    organizer: "Khoa CNTT",
    role: "participant",
  },
  seminar: {
    title: "Seminar Học thuật về Nghiên cứu AI và Machine Learning",
    description: "Buổi chia sẻ chuyên đề nghiên cứu khoa học sinh viên và định hướng khóa luận tốt nghiệp.",
    format: "seminar, workshop",
    organizer: "Khoa CNTT",
    role: "participant",
  },
  volunteer: {
    title: "Chiến dịch Tình nguyện Mùa Hè Xanh 2026 (3 ngày)",
    description: "Hoạt động tình nguyện vì cộng đồng kéo dài 3 ngày tại xã vùng ven.",
    format: "volunteer, campaign",
    organizer: "Đoàn Thanh niên - Hội Sinh viên HCMUS",
    role: "participant",
  },
  blood: {
    title: "Ngày hội Hiến máu tình nguyện Giọt hồng HCMUS",
    description: "Chương trình hiến máu nhân đạo vì cộng đồng do Hội Sinh viên Trường tổ chức.",
    format: "volunteer, donation",
    organizer: "HSV",
    role: "participant",
  },
  mobilize: {
    title: "Huy động lực lượng cổ vũ Giải Bóng đá Futsal sinh viên ĐHQG-HCM",
    description: "Thông báo huy động sinh viên tham gia cổ vũ trận đấu cấp Trường.",
    format: "sports, cheering",
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
  // Tab 3 (Scoring P5-P6) & Tab 4 (MMR Selector P7) state
  recBootstrap: null,
  recLastResult: null,
  recActivePersonaId: "balanced_cs1",
  recSubTab: "ranked", // "ranked" (pre-MMR Scoring) | "dropped"
  recDropReasonFilter: "ALL",
  recParticipatedIds: [],
  recHiddenIds: [],
  recWeeklySlots: [],
  recBusyIntervals: [],
  mmrLastResult: null,
  mmrSubTab: "selected", // "selected" (P7 MMR) | "ranked" (pre-MMR P6 comparison)
  mmrNeedsRefresh: true,
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
            <h3 class="card-subtitle">Query Vectors (3 profiles · ${escapeHtml(res.embedder_status?.active_model || "e5-small")})</h3>
            <span class="mono text-muted" style="font-size:11.5px">${res.vector_dimension}d · ||v||₂ = ${res.vector_l2_norm} · ${res.reused_cached_query ? "cached" : "live"}</span>
          </div>
          <div class="badge-row" style="margin:6px 0 8px">
            <span class="badge badge-teal mono">Sample → ${escapeHtml((db.vector_routing_used && db.vector_routing_used.sample_stage1) || params.sample_branch_vector || "title_description")}</span>
            <span class="badge badge-orange mono">Rubric → ${escapeHtml((db.vector_routing_used && db.vector_routing_used.rubric_stage1) || params.rubric_branch_vector || "title_description")}</span>
          </div>
          ${
            res.profile_query_texts
              ? ["title", "title_description", "title_description_format"]
                  .map((pname) => {
                    const sProf = (db.vector_routing_used && db.vector_routing_used.sample_stage1) || params.sample_branch_vector || "title_description";
                    const rProf = (db.vector_routing_used && db.vector_routing_used.rubric_stage1) || params.rubric_branch_vector || "title_description";
                    const tags = [];
                    if (sProf === pname) tags.push("Sample");
                    if (rProf === pname) tags.push("Rubric");
                    const activeTag = tags.length ? ` [${tags.join(" + ")}]` : "";
                    return `<div class="query-preview mono" style="margin-top:4px"><strong>${escapeHtml(pname)}${escapeHtml(activeTag)}:</strong> ${escapeHtml(res.profile_query_texts[pname] || "")}</div>`;
                  })
                  .join("")
              : `<div class="query-preview mono">${escapeHtml(res.formatted_query_text)}</div>`
          }
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
  if ($("#paramSampleVec")) {
    $("#paramSampleVec").value = dp.sample_branch_vector || "title_description";
  }
  if ($("#paramRubricVec")) {
    $("#paramRubricVec").value = dp.rubric_branch_vector || "title_description";
  }
  if ($("#batchSampleVec")) {
    $("#batchSampleVec").value = dp.sample_branch_vector || "title_description";
  }
  if ($("#batchRubricVec")) {
    $("#batchRubricVec").value = dp.rubric_branch_vector || "title_description";
  }
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
      format: $("#clsFormat") ? $("#clsFormat").value : "",
      organizer: $("#clsOrg").value,
      role: $("#clsRole").value,
      sample_branch_vector: $("#paramSampleVec") ? $("#paramSampleVec").value : "title_description",
      rubric_branch_vector: $("#paramRubricVec") ? $("#paramRubricVec").value : "title_description",
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
  const navLbl = $("#navEventsLabel");
  if (navLbl) {
    navLbl.textContent = `Sự kiện DB (${totalCount})`;
  }
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
      format: ev.format || [],
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
   TAB 3: RECOMMENDATION DEMO (P4 Demo + Demo M4/M6 + P5 + P6)
   ========================================================================= */

const WEEKDAY_LABELS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"];

function fmtHour(h) {
  const mins = Math.round(Number(h) * 60);
  const hh = String(Math.floor(mins / 60)).padStart(2, "0");
  const mm = String(mins % 60).padStart(2, "0");
  return `${hh}:${mm}`;
}

function fmtShortDateTime(iso) {
  if (!iso) return "Chưa rõ thời gian";
  const m = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})/);
  if (!m) return String(iso);
  return `${m[3]}/${m[2]}/${m[1]} ${m[4]}`;
}

function updateRecCurrentTotalBadge() {
  const c1 = Math.min(30, Math.max(0, Number($("#recPtsC1").value || 0)));
  const c2 = Math.min(20, Math.max(0, Number($("#recPtsC2").value || 0)));
  const c3 = Math.min(25, Math.max(0, Number($("#recPtsC3").value || 0)));
  const c4 = Math.min(25, Math.max(0, Number($("#recPtsC4").value || 0)));
  const c5 = Math.min(30, Math.max(0, Number($("#recPtsC5").value || 0)));
  const total = Math.min(100, c1 + c2 + c3 + c4 + c5);
  const target = Number($("#recTargetTotal").value || 85);
  $("#recCurrentTotalBadge").textContent = `${total} / ${target}đ`;
}

function renderRecSlotsList() {
  const container = $("#recSlotsList");
  const items = [];
  state.recWeeklySlots.forEach((s, idx) => {
    const day = WEEKDAY_LABELS[Number(s.day_of_week) % 7] || "T?";
    items.push(`
      <div class="slot-pill">
        <span><strong class="mono">${day} ${fmtHour(s.start_hour)}–${fmtHour(s.end_hour)}</strong> · ${escapeHtml(s.label || "Lịch học")}</span>
        <button type="button" class="slot-remove-btn" data-remove-slot="${idx}" title="Xóa slot">×</button>
      </div>
    `);
  });
  state.recBusyIntervals.forEach((b, idx) => {
    items.push(`
      <div class="slot-pill">
        <span><span class="badge badge-danger mono" style="padding:1px 6px">${b.is_exam ? "THI" : "BẬN"}</span> ${escapeHtml(b.label || "Lịch cố định")} <span class="mono text-muted">(${fmtShortDateTime(b.start_datetime)})</span></span>
        <button type="button" class="slot-remove-btn" data-remove-busy="${idx}" title="Xóa lịch">×</button>
      </div>
    `);
  });
  container.innerHTML = items.length
    ? items.join("")
    : `<div class="helper-text">Chưa có khung giờ bận nào.</div>`;
}

function renderRecUserActionState() {
  const box = $("#recUserActionState");
  if (!state.recParticipatedIds.length && !state.recHiddenIds.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  box.innerHTML = `
    <div class="card-title-row">
      <strong>Trạng thái test P5 đang bật:</strong>
      <button id="btnRecClearActions" type="button" class="btn btn-ghost btn-sm" style="padding:2px 8px">Xóa hết</button>
    </div>
    <div class="badge-row" style="margin-top:6px">
      ${state.recParticipatedIds
        .map(
          (id) => `<span class="badge badge-teal mono">Đã tham gia: ${escapeHtml(id)}</span>`
        )
        .join("")}
      ${state.recHiddenIds
        .map((id) => `<span class="badge badge-orange mono">Đã ẩn: ${escapeHtml(id)}</span>`)
        .join("")}
    </div>
  `;
}

function applyCreditPresetToForm(presetId) {
  const presets = (state.recBootstrap && state.recBootstrap.credit_presets) || [];
  const p = presets.find((x) => x.id === presetId);
  if (!p) return;
  const g = p.current_by_group || {};
  $("#recPtsC1").value = g.C1 ?? 0;
  $("#recPtsC2").value = g.C2 ?? 0;
  $("#recPtsC3").value = g.C3 ?? 0;
  $("#recPtsC4").value = g.C4 ?? 0;
  $("#recPtsC5").value = g.C5 ?? 0;
  $("#recTargetTotal").value = p.target_total ?? 85;
  updateRecCurrentTotalBadge();
}

function applySchedulePresetToForm(presetId) {
  const presets = (state.recBootstrap && state.recBootstrap.schedule_presets) || [];
  const p = presets.find((x) => x.id === presetId);
  if (!p) return;
  state.recWeeklySlots = (p.weekly_slots || []).map((s) => ({ ...s }));
  state.recBusyIntervals = (p.busy_intervals || []).map((b) => ({ ...b }));
  const ew = (p.exam_weeks && p.exam_weeks[0]) || ["", ""];
  $("#recExamStart").value = ew[0] || "";
  $("#recExamEnd").value = ew[1] || "";
  $("#recScheduleSummaryBadge").textContent = presetId;
  renderRecSlotsList();
}

function applyWeightPresetToForm(presetId) {
  const map = (state.recBootstrap && state.recBootstrap.weight_presets) || {};
  const p = map[presetId];
  if (!p || !p.weights) return;
  $("#recW1").value = p.weights.w1_gain;
  $("#recW2").value = p.weights.w2_interest;
  $("#recW3").value = p.weights.w3_fit;
  $("#recW4").value = p.weights.w4_urgency;
  $("#recWeightSummaryBadge").textContent = presetId;
}

function applyRecPersona(personaId) {
  const personas = (state.recBootstrap && state.recBootstrap.personas) || [];
  const p = personas.find((x) => x.id === personaId);
  if (!p) return;
  state.recActivePersonaId = personaId;
  state.recParticipatedIds = [];
  state.recHiddenIds = [];

  document.querySelectorAll("#recPersonaBar .persona-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.personaId === personaId);
  });

  // Interest chips
  const selectedChips = new Set(p.interest_chips || []);
  document.querySelectorAll("#recChipsGrid .interest-toggle").forEach((lbl) => {
    const inp = lbl.querySelector("input");
    const checked = selectedChips.has(inp.value);
    inp.checked = checked;
    lbl.classList.toggle("checked", checked);
  });
  $("#recFreeText").value = p.free_text || "";

  // Constraints
  $("#recRefNow").value = String(p.reference_now || "2026-10-01").slice(0, 10);
  $("#recHomeCampus").value = p.home_campus || "cs1_nvc";
  $("#recCohort").value = p.cohort || "";
  $("#recMaxTravel").value = p.max_travel_min ?? 45;
  $("#recMaxBudget").value = p.max_budget_vnd ?? "";
  $("#recPrefStart").value = (p.pref_hours && p.pref_hours[0]) ?? 8;
  $("#recPrefEnd").value = (p.pref_hours && p.pref_hours[1]) ?? 21;
  $("#recStrictTravel").checked = Boolean(p.strict_max_travel);
  $("#recIncludeMandatory").checked = Boolean(p.include_mandatory_demo);

  // Demo M6
  $("#recCreditPreset").value = p.credit_preset || "mid_semester";
  applyCreditPresetToForm(p.credit_preset || "mid_semester");

  // Demo M4
  $("#recSchedulePreset").value = p.schedule_preset || "light_schedule";
  applySchedulePresetToForm(p.schedule_preset || "light_schedule");

  // Weights
  $("#recWeightPreset").value = p.weight_preset || "balanced";
  applyWeightPresetToForm(p.weight_preset || "balanced");

  renderRecUserActionState();
}

function updateRecMmrSummaryBadge() {
  const badge = $("#recMmrSummaryBadge");
  if (!badge) return;
  const topN = Number($("#recMmrTopN")?.value || 5);
  const lam = Number($("#recMmrLambda")?.value ?? 0.7);
  badge.textContent = `Top-${topN} · λ=${lam.toFixed(2)}`;
  badge.className = "badge badge-teal mono";
}

function applyMmrDefaultsToForm(defaults) {
  const d = defaults || (state.recBootstrap && state.recBootstrap.mmr_defaults) || {};
  if ($("#recMmrTopN")) $("#recMmrTopN").value = d.top_n ?? 5;
  if ($("#recMmrLambda")) $("#recMmrLambda").value = d.lambda_param ?? 0.7;
  if ($("#recMmrAlpha")) $("#recMmrAlpha").value = d.alpha_semantic ?? 0.7;
  if ($("#recMmrCosineMode")) $("#recMmrCosineMode").value = d.cosine_rescale_mode || "pool_minmax";
  if ($("#recMmrExplore")) $("#recMmrExplore").checked = d.enable_exploration_slot !== false;
  if ($("#recMmrPinMandatory")) $("#recMmrPinMandatory").checked = d.pin_mandatory !== false;
  updateRecMmrSummaryBadge();
}

function buildRecPayloadFromForm({ includeMmr = false } = {}) {
  const chips = Array.from(document.querySelectorAll("#recChipsGrid input:checked")).map(
    (inp) => inp.value
  );
  const examS = $("#recExamStart").value.trim();
  const examE = $("#recExamEnd").value.trim();
  const examWeeks = examS && examE ? [[examS, examE]] : [];

  const weightPreset = $("#recWeightPreset").value;
  const customWeights = {
    w1_gain: Number($("#recW1").value || 0),
    w2_interest: Number($("#recW2").value || 0),
    w3_fit: Number($("#recW3").value || 0),
    w4_urgency: Number($("#recW4").value || 0),
  };

  const budgetRaw = $("#recMaxBudget").value.trim();
  const travelRaw = $("#recMaxTravel").value.trim();

  const payload = {
    data_source: "database",
    user_id: state.recActivePersonaId || "demo_user",
    cohort: $("#recCohort").value.trim() || null,
    interest_chips: chips,
    free_text: $("#recFreeText").value.trim(),
    home_campus: $("#recHomeCampus").value,
    max_travel_min: travelRaw === "" ? null : Number(travelRaw),
    strict_max_travel: $("#recStrictTravel").checked,
    max_budget_vnd: budgetRaw === "" ? null : Number(budgetRaw),
    pref_hours: [Number($("#recPrefStart").value || 8), Number($("#recPrefEnd").value || 21)],
    participated_event_ids: [...state.recParticipatedIds],
    hidden_event_ids: [...state.recHiddenIds],
    reference_now: `${$("#recRefNow").value || "2026-10-01"}T08:00:00+07:00`,
    include_mandatory_demo: $("#recIncludeMandatory").checked,
    credit_preset: $("#recCreditPreset").value,
    current_by_group: {
      C1: Number($("#recPtsC1").value || 0),
      C2: Number($("#recPtsC2").value || 0),
      C3: Number($("#recPtsC3").value || 0),
      C4: Number($("#recPtsC4").value || 0),
      C5: Number($("#recPtsC5").value || 0),
    },
    target_total: Number($("#recTargetTotal").value || 85),
    schedule_preset: $("#recSchedulePreset").value,
    schedule: {
      preset_id: $("#recSchedulePreset").value,
      weekly_slots: state.recWeeklySlots,
      busy_intervals: state.recBusyIntervals,
      exam_weeks: examWeeks,
      default_travel_buffer_min: 30,
    },
    weight_preset: weightPreset,
    custom_weights: customWeights,
    interest_calibration_mode: $("#recInterestMode").value,
    mmr: includeMmr
      ? {
          enabled: true,
          top_n: Math.max(1, Number($("#recMmrTopN")?.value || 5)),
          lambda_param: Number($("#recMmrLambda")?.value ?? 0.7),
          alpha_semantic: Number($("#recMmrAlpha")?.value ?? 0.7),
          cosine_rescale_mode: $("#recMmrCosineMode") ? $("#recMmrCosineMode").value : "pool_minmax",
          enable_exploration_slot: $("#recMmrExplore") ? $("#recMmrExplore").checked : true,
          pin_mandatory: $("#recMmrPinMandatory") ? $("#recMmrPinMandatory").checked : true,
        }
      : {
          enabled: false,
        },
  };
  return payload;
}

function renderRecEventCard(ev, { cal, isMmrView }) {
  const c = ev.components || {};
  const wc = ev.weighted_contributions || {};
  const ce = ev.credit_estimate || {};
  const fa = ev.fit_assessment || {};
  const sub = fa.sub_scores || {};
  const gd = ev.gain_details || {};
  const st = ev.selection_trace || null;
  const isExplore = Boolean(isMmrView && (ev.is_exploration_slot || st?.slot_type === "exploration"));
  const barDenom = Math.max(1.0, Number(ev.score_total || 1.0));
  const pct = (v) => Math.max(0, Math.min(100, (Number(v || 0) / barDenom) * 100)).toFixed(1);

  const ptsBadge = ev.is_mandatory
    ? `<span class="badge badge-danger mono">BẮT BUỘC · ${escapeHtml(ev.rubric_item_id || "6.4")} (tránh trừ ${Math.abs(ev.mandatory_info?.penalty_if_missed || 20)}đ)</span>`
    : `<span class="badge badge-teal mono">${escapeHtml(ev.rubric_item_id || "?")} (${escapeHtml(ev.parent_group_id || "?")}) · +${ce.effective_points ?? 0}đ</span>`;

  let slotBadgesHtml = "";
  if (isMmrView && st) {
    if (st.slot_type === "exploration") {
      slotBadgesHtml += `<span class="badge badge-purple mono">KHÁM PHÁ · SEL_EXPLORATION</span>`;
    } else if (st.slot_type === "mandatory") {
      slotBadgesHtml += `<span class="badge badge-danger mono">GHIM BẮT BUỘC</span>`;
    } else {
      slotBadgesHtml += `<span class="badge badge-teal mono">MMR</span>`;
    }
    const delta = Number(st.rank_delta || 0);
    if (delta > 0) {
      slotBadgesHtml += `<span class="badge rank-shift-up mono">↑ +${delta} bậc (P6 #${st.p6_rank})</span>`;
    } else if (delta < 0) {
      slotBadgesHtml += `<span class="badge rank-shift-down mono">↓ ${delta} bậc (P6 #${st.p6_rank})</span>`;
    } else {
      slotBadgesHtml += `<span class="badge badge-muted mono">P6 #${st.p6_rank}</span>`;
    }
  }

  const modeLabel =
    ev.online_mode === "online"
      ? "Online · 0p"
      : `${ev.online_mode || "offline"} · ~${fa.estimated_travel_min ?? "?"}p`;

  const feeLabel =
    ev.fee_vnd === 0
      ? "Miễn phí"
      : ev.fee_vnd != null
        ? `${Number(ev.fee_vnd).toLocaleString("vi-VN")}đ`
        : "Không rõ phí";

  const deadlineLabel =
    ev.days_to_deadline != null ? `Còn ${fmtOptNum(ev.days_to_deadline, 1)} ngày` : "Không rõ hạn";

  const reasoning = ev.reasoning || {};
  const rCriteria = reasoning.criteria || [];
  const verdictClass = (v) => {
    if (v === "bắt buộc") return "verdict-danger";
    if (v === "cao") return "verdict-high";
    if (v === "khá") return "verdict-good";
    if (v === "trung bình" || v === "trung tính") return "verdict-mid";
    return "verdict-low";
  };

  const reasoningRowsHtml = rCriteria
    .map((rc) => {
      const barPct = Math.max(0, Math.min(100, Number(rc.raw_score || 0) * 100)).toFixed(0);
      const mathExpr =
        rc.weight !== null && rc.weight !== undefined
          ? `${fmtOptNum(rc.weight, 2)} × ${fmtOptNum(rc.raw_score, 3)} = +${fmtOptNum(rc.weighted_score, 3)}`
          : `+${fmtOptNum(rc.weighted_score, 3)} (trực tiếp)`;
      return `
        <div class="reasoning-row">
          <div class="reasoning-crit">
            <div class="reasoning-crit-head">
              <span class="comp-dot seg-${escapeHtml(rc.key)}"></span>
              <strong>${escapeHtml(rc.label)}</strong>
              <span class="verdict-tag ${verdictClass(rc.verdict)}">${escapeHtml(rc.verdict)}</span>
            </div>
            <div class="mini-bar-row">
              <div class="mini-bar-track"><div class="mini-bar-fill seg-${escapeHtml(rc.key)}" style="width:${barPct}%"></div></div>
              <span class="mono reasoning-math">${escapeHtml(mathExpr)} <span class="text-muted">(${fmtOptNum(rc.share_pct, 0)}%)</span></span>
            </div>
          </div>
          <div class="reasoning-text">${escapeHtml(rc.reason)}</div>
        </div>
      `;
    })
    .join("");

  const mmrTraceHtml =
    isMmrView && st && st.selection_reason
      ? `
        <div class="rec-mmr-trace-box ${st.slot_type === "exploration" ? "exploration-trace" : st.slot_type === "mandatory" ? "mandatory-trace" : ""}">
          <strong>P7 · ${st.slot_type === "exploration" ? "Khám phá" : st.slot_type === "mandatory" ? "Ghim bắt buộc" : "MMR"}:</strong>
          ${escapeHtml(st.selection_reason)}
        </div>
      `
      : "";

  const scoreSubLine =
    isMmrView && st
      ? st.mmr_score !== null && st.mmr_score !== undefined
        ? `<div class="mono text-muted" style="font-size:11px">MMR ${fmtOptNum(st.mmr_score, 3)} · Sim ${fmtOptNum(st.max_redundancy, 2)}</div>`
        : st.explore_score !== null && st.explore_score !== undefined
          ? `<div class="mono text-muted" style="font-size:11px">Exp ${fmtOptNum(st.explore_score, 3)} · Sim ${fmtOptNum(st.max_redundancy, 2)}</div>`
          : ""
      : "";

  const cardClasses = [
    "rec-card",
    ev.is_mandatory ? "mandatory-card" : "",
    isExplore ? "exploration-card" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return `
    <div class="${cardClasses}">
      <div class="rec-card-top">
        <div>
          <div class="badge-row" style="margin-bottom:6px">
            <span class="rec-rank-pill mono">#${ev.rank}</span>
            ${slotBadgesHtml}
            ${ptsBadge}
            ${fa.in_exam_week ? `<span class="badge badge-orange">Tuần thi</span>` : ""}
            <span class="badge badge-muted mono">${escapeHtml(ev.event_id)}</span>
          </div>
          <div style="font-size:15px;font-weight:700;color:var(--color-fg)">${escapeHtml(ev.title)}</div>
          <div class="rec-meta-line" style="margin-top:4px">
            <span>${fmtShortDateTime(ev.start_datetime)}</span>
            <span>• ${escapeHtml(modeLabel)}</span>
            <span>• ${escapeHtml(feeLabel)}</span>
            <span>• ${escapeHtml(deadlineLabel)}</span>
          </div>
          ${ev.description ? `<p class="rec-event-desc">${escapeHtml(ev.description)}</p>` : ""}
        </div>
        <div class="rec-score-box">
          <div class="kpi-label">P6 Score</div>
          <div class="rec-score-total mono">${fmtOptNum(ev.score_total, 3)}</div>
          ${scoreSubLine}
        </div>
      </div>

      ${mmrTraceHtml}

      ${
        reasoning.headline
          ? `<div class="rec-reasoning-headline ${ev.is_mandatory ? "mandatory-headline" : ""}">${escapeHtml(reasoning.headline)}</div>`
          : ""
      }

      <div class="score-bar-track" title="Đóng góp theo trọng số: Gain / Interest / Fit / Urgency / Penalty">
        <div class="seg-gain" style="width:${pct(wc.gain)}%"></div>
        <div class="seg-interest" style="width:${pct(wc.interest)}%"></div>
        <div class="seg-fit" style="width:${pct(wc.fit)}%"></div>
        <div class="seg-urgency" style="width:${pct(wc.urgency)}%"></div>
        <div class="seg-penalty" style="width:${pct(wc.penalty)}%"></div>
      </div>

      <div class="component-pills">
        <div class="comp-pill">
          <div class="comp-pill-label"><span class="comp-dot seg-gain"></span>Gain</div>
          <div class="comp-pill-val mono">${fmtOptNum(c.gain, 3)} <span class="text-muted" style="font-size:10.5px">(+${fmtOptNum(wc.gain, 3)})</span></div>
        </div>
        <div class="comp-pill">
          <div class="comp-pill-label"><span class="comp-dot seg-interest"></span>Interest</div>
          <div class="comp-pill-val mono">${fmtOptNum(c.interest, 3)} <span class="text-muted" style="font-size:10.5px">(+${fmtOptNum(wc.interest, 3)})</span></div>
        </div>
        <div class="comp-pill">
          <div class="comp-pill-label"><span class="comp-dot seg-fit"></span>Fit</div>
          <div class="comp-pill-val mono">${fmtOptNum(c.fit, 3)} <span class="text-muted" style="font-size:10.5px">(+${fmtOptNum(wc.fit, 3)})</span></div>
        </div>
        <div class="comp-pill">
          <div class="comp-pill-label"><span class="comp-dot seg-urgency"></span>Urgency</div>
          <div class="comp-pill-val mono">${fmtOptNum(c.urgency, 3)} <span class="text-muted" style="font-size:10.5px">(+${fmtOptNum(wc.urgency, 3)})</span></div>
        </div>
        <div class="comp-pill">
          <div class="comp-pill-label"><span class="comp-dot seg-penalty"></span>Penalty</div>
          <div class="comp-pill-val mono">${Number(c.penalty) > 0 ? "+" : ""}${fmtOptNum(c.penalty, 3)}</div>
        </div>
      </div>

      <div class="rec-card-actions">
        <details style="flex:1 1 100%" ${ev.rank === 1 ? "open" : ""}>
          <summary class="helper-text" style="cursor:pointer;font-weight:600">Chi tiết tính toán &amp; Giải thích tiêu chí (Reasoning · Kiểm chứng trọng số)</summary>
          <div class="reasoning-panel">
            ${reasoningRowsHtml}
            ${
              reasoning.formula_check
                ? `<div class="formula-audit-bar mono"><strong>Kiểm chứng tổng:</strong> ${escapeHtml(reasoning.formula_check)}</div>`
                : ""
            }
            <div class="raw-params-strip mono">
              <span><strong>M6:</strong> P_eff=${ce.effective_points}, P_use=${gd.p_use ?? 0}, m_gap=${fmtOptNum(gd.m_gap, 3)}, src=${escapeHtml(ce.source)}</span>
              <span><strong>E5:</strong> cos=${fmtOptNum(ev.interest_cos_raw, 4)} trong [${fmtOptNum(cal.c_min, 3)}, ${fmtOptNum(cal.c_max, 3)}]</span>
              <span><strong>M4:</strong> f_tr=${fmtOptNum(sub.f_travel, 2)}, f_sl=${fmtOptNum(sub.f_slack, 2)}, f_pr=${fmtOptNum(sub.f_pref, 2)}, f_ex=${fmtOptNum(sub.f_exam, 2)}</span>
              ${
                st && st.most_similar_selected_id
                  ? `<span><strong>P7 Sim:</strong> max_red=${fmtOptNum(st.max_redundancy, 3)} vs ${escapeHtml(st.most_similar_selected_id)} (cos=${fmtOptNum(st.raw_cosine_with_most_similar, 3)}, sem=${fmtOptNum(st.semantic_sim_with_most_similar, 3)}, rub=${fmtOptNum(st.rubric_sim_with_most_similar, 2)})</span>`
                  : ""
              }
              <span><strong>Địa điểm:</strong> ${escapeHtml(ev.location || "Chưa rõ")}</span>
              ${
                ev.rec_embedding_text
                  ? `<span style="flex:1 1 100%;word-break:break-word"><strong>4-Field Event Text:</strong> ${escapeHtml(ev.rec_embedding_text)}</span>`
                  : ""
              }
            </div>
          </div>
        </details>
        <div class="badge-row" style="margin-top:4px">
          <button type="button" class="btn btn-ghost btn-sm" data-rec-join="${escapeHtml(ev.event_id)}">Đã tham gia</button>
          <button type="button" class="btn btn-ghost btn-sm" data-rec-hide="${escapeHtml(ev.event_id)}">Ẩn</button>
        </div>
      </div>
    </div>
  `;
}

function renderMmrProfileSummary() {
  const box = $("#mmrProfileSummary");
  if (!box) return;
  const res = state.recLastResult || state.mmrLastResult;
  if (!res) {
    box.innerHTML = `Chưa có dữ liệu Scoring P6. Hãy chạy Scoring hoặc bấm <strong>Chạy Chọn lọc MMR (P7)</strong>.`;
    return;
  }
  const sum = res.pipeline_summary || {};
  const w = (res.profile_used && res.profile_used.weights) || {};
  const m6 = res.demo_module6_state || {};
  box.innerHTML = `
    <div class="badge-row" style="margin-bottom:6px">
      <span class="badge badge-teal mono">drl_database.db · ${sum.eligible_after_p5 ?? 0} ứng viên P6</span>
      <span class="badge badge-muted mono">Mốc: ${fmtShortDateTime(sum.reference_now)}</span>
    </div>
    <div class="mono" style="font-size:11.5px">
      ĐRL (Demo M6): <strong>${m6.current_total ?? 0}/${m6.target_total ?? 85}đ</strong> ·
      w=(${fmtOptNum(w.w1_gain, 2)}, ${fmtOptNum(w.w2_interest, 2)}, ${fmtOptNum(w.w3_fit, 2)}, ${fmtOptNum(w.w4_urgency, 2)})
    </div>
  `;
}

function renderRecommendationResults() {
  const res = state.recLastResult;
  const pane = $("#recResultPane");
  if (!res) {
    pane.innerHTML = `<div class="card empty-state">Bấm <strong>Chạy Scoring (P5–P6)</strong> để xem bảng xếp hạng chưa MMR.</div>`;
    return;
  }

  const sum = res.pipeline_summary || {};
  const cal = res.interest_calibration || {};
  const m6 = res.demo_module6_state || {};
  const ranked = res.ranked_events || [];
  const dropped = res.dropped_events || [];
  const reasonCounts = sum.drop_reason_counts || {};
  const embFields = (cal.event_embedding_fields || ["title", "description", "topic", "format"]).join(" + ");

  const kpiHtml = `
    <div class="card">
      <div class="card-title-row">
        <h2 class="card-title">Danh sách Scoring P6 · Chưa áp dụng MMR (P4 → Demo M4/M6 → P5 → P6)</h2>
        <div class="badge-row">
          <span class="mono text-muted" style="font-size:12px">${escapeHtml(res.elapsed_ms)} ms · mốc ${fmtShortDateTime(sum.reference_now)}</span>
          <button type="button" class="btn btn-primary btn-sm" data-open-mmr="1">Mở Bộ chọn MMR (P7) →</button>
        </div>
      </div>
      <div class="badge-row" style="margin-top:8px">
        <span class="badge badge-teal mono">Nguồn: drl_database.db (${sum.total_batch_events ?? 0} sự kiện)</span>
        <span class="badge badge-muted mono">Event vec (${escapeHtml((cal.event_embedding_prefix || "passage: ").trim())}): ${escapeHtml(embFields)}</span>
        <span class="badge badge-muted mono">Module 6: DemoCreditProvider</span>
      </div>
      <div class="rec-kpi-grid" style="margin-top:12px">
        <div class="kpi-box">
          <div class="kpi-label">Đầu vào PREDICTED (DB)</div>
          <div class="kpi-val mono text-teal">${sum.predicted_events_entered_p5} <span class="text-muted" style="font-size:12px">/ ${sum.total_batch_events}</span></div>
          <div class="helper-text" style="font-size:11px">Bỏ ${sum.excluded_not_predicted} chưa PREDICTED</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Đạt P5 (Xếp hạng P6)</div>
          <div class="kpi-val mono text-teal">${sum.eligible_after_p5}</div>
          <div class="helper-text" style="font-size:11px">Bị lọc cứng P5: ${sum.dropped_by_p5}</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Interest [c_min .. c_max]</div>
          <div class="kpi-val mono" style="font-size:13.5px">[${fmtOptNum(cal.c_min, 3)}, ${fmtOptNum(cal.c_max, 3)}]</div>
          <div class="helper-text" style="font-size:11px">${cal.sample_count} sample · ${escapeHtml(cal.mode)}</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">ĐRL hiện tại (Demo M6)</div>
          <div class="kpi-val mono">${m6.current_total ?? 0} / ${m6.target_total ?? 85}đ</div>
          <div class="helper-text" style="font-size:11px">Dư C1:${m6.remaining_cap_by_group?.C1 ?? 0} · C3:${m6.remaining_cap_by_group?.C3 ?? 0} · C4:${m6.remaining_cap_by_group?.C4 ?? 0}</div>
        </div>
      </div>

      <div class="rec-subtabs" style="margin-top:14px">
        <div class="status-filter-pills">
          <button type="button" class="filter-pill ${state.recSubTab === "ranked" ? "active" : ""}" data-rec-subtab="ranked">
            Xếp hạng Scoring P6 · Chưa MMR (${ranked.length})
          </button>
          <button type="button" class="filter-pill ${state.recSubTab === "dropped" ? "active" : ""}" data-rec-subtab="dropped">
            Bị lọc cứng P5 · Vì sao bị loại? (${dropped.length})
          </button>
        </div>
        <span class="mono text-muted" style="font-size:11.5px">
          w=(${fmtOptNum(res.profile_used?.weights?.w1_gain, 2)}, ${fmtOptNum(res.profile_used?.weights?.w2_interest, 2)}, ${fmtOptNum(res.profile_used?.weights?.w3_fit, 2)}, ${fmtOptNum(res.profile_used?.weights?.w4_urgency, 2)})
        </span>
      </div>
    </div>
  `;

  let bodyHtml = "";
  if (state.recSubTab === "ranked") {
    if (!ranked.length) {
      bodyHtml =
        Number(sum.total_batch_events || 0) === 0
          ? `<div class="card empty-state"><strong>Database <code>drl_database.db</code> hiện đang trống (0 sự kiện trong bảng <code>events</code>).</strong></div>`
          : `<div class="card empty-state">Không có sự kiện nào vượt qua bộ lọc cứng P5. Hãy xem tab <strong>Bị lọc cứng P5</strong> hoặc nới lỏng ràng buộc (mốc ngày, ngân sách, trần C1, lịch học).</div>`;
    } else {
      bodyHtml = ranked.map((ev) => renderRecEventCard(ev, { cal, isMmrView: false })).join("");
    }
  } else {
    const reasonKeys = Object.keys(reasonCounts).sort((a, b) => reasonCounts[b] - reasonCounts[a]);
    const filteredDrops =
      state.recDropReasonFilter === "ALL"
        ? dropped
        : dropped.filter((d) =>
            (d.reasons || []).some((r) => r.code === state.recDropReasonFilter)
          );

    const pillsHtml = `
      <div class="card" style="padding:12px 16px">
        <div class="status-filter-pills">
          <button type="button" class="filter-pill ${state.recDropReasonFilter === "ALL" ? "active" : ""}" data-drop-filter="ALL">
            Tất cả (${dropped.length})
          </button>
          ${reasonKeys
            .map(
              (code) => `
                <button type="button" class="filter-pill mono ${state.recDropReasonFilter === code ? "active" : ""}" data-drop-filter="${escapeHtml(code)}">
                  ${escapeHtml(code)} (${reasonCounts[code]})
                </button>
              `
            )
            .join("")}
        </div>
      </div>
    `;

    const listHtml = filteredDrops.length
      ? filteredDrops
          .map(
            (d) => `
              <div class="dropped-card">
                <div class="card-title-row">
                  <div>
                    <span class="mono text-muted" style="font-size:11.5px">${escapeHtml(d.event_id)} · ${escapeHtml(d.rubric_item_id || "?")} (${escapeHtml(d.parent_group_id || "?")})</span>
                    <div style="font-weight:700;font-size:13.5px">${escapeHtml(d.title)}</div>
                  </div>
                  <div class="badge-row">
                    ${(d.reasons || [])
                      .map((r) => `<span class="badge badge-danger mono">${escapeHtml(r.code)}</span>`)
                      .join("")}
                  </div>
                </div>
                ${
                  d.description
                    ? `<p class="dropped-event-desc">${escapeHtml(d.description)}</p>`
                    : ""
                }
                <ul class="drop-reason-list">
                  ${(d.reasons || []).map((r) => `<li>${escapeHtml(r.message)}</li>`).join("")}
                </ul>
              </div>
            `
          )
          .join("")
      : `<div class="card empty-state">Không có sự kiện bị lọc khớp mã lý do này.</div>`;

    bodyHtml = pillsHtml + listHtml;
  }

  pane.innerHTML = kpiHtml + bodyHtml;
  renderMmrProfileSummary();
}

function renderMmrResults() {
  const res = state.mmrLastResult;
  const pane = $("#mmrResultPane");
  if (!pane) return;
  if (!res) {
    pane.innerHTML = `<div class="card empty-state">Bấm <strong>Chạy Chọn lọc MMR (P7)</strong> để đa dạng hóa danh sách từ bảng Scoring P6.</div>`;
    return;
  }

  const sum = res.pipeline_summary || {};
  const cal = res.interest_calibration || {};
  const mmr = res.mmr_selection || null;
  const mmrCfg = (mmr && mmr.config) || {};
  const mmrMetrics = (mmr && mmr.metrics) || {};
  const simCal = (mmr && mmr.similarity_calibration) || {};
  const selected = res.selected_events || [];
  const ranked = res.ranked_events || [];

  const ildGain = Number(mmrMetrics.ild_gain || 0);
  const ildGainTxt = `${ildGain >= 0 ? "+" : ""}${fmtOptNum(ildGain, 3)}`;
  const mmrGroupsTxt = (mmrMetrics.mmr_groups || []).join(", ") || "—";
  const p6GroupsTxt = (mmrMetrics.p6_topn_groups || []).join(", ") || "—";
  const simRangeTxt =
    simCal.raw_cos_min != null && simCal.raw_cos_max != null
      ? `[${fmtOptNum(simCal.raw_cos_min, 3)}, ${fmtOptNum(simCal.raw_cos_max, 3)}]`
      : "—";

  const mmrBannerHtml = mmr
    ? `
      <div class="rec-mmr-banner" style="margin-top:10px">
        <div>
          <div class="kpi-label">P7 · Chọn danh sách MMR</div>
          <div class="kpi-val mono text-teal" style="font-size:13.5px">
            Top-${selected.length} (λ=${fmtOptNum(mmrCfg.lambda_param, 2)}, α=${fmtOptNum(mmrCfg.alpha_semantic, 2)})
          </div>
          <div class="helper-text" style="font-size:11px">
            Ghim: ${mmr.mandatory_count} · MMR: ${mmr.relevance_count} · Khám phá: ${mmr.exploration_count}
          </div>
        </div>
        <div>
          <div class="kpi-label">Đa dạng ILD (MMR vs P6)</div>
          <div class="kpi-val mono" style="font-size:13.5px">
            ${fmtOptNum(mmrMetrics.mmr_ild, 3)} <span class="text-muted" style="font-size:11.5px">vs ${fmtOptNum(mmrMetrics.p6_topn_ild, 3)}</span>
          </div>
          <div class="helper-text mono" style="font-size:11px;color:${ildGain >= 0 ? "var(--color-primary-hover)" : "#9A3412"}">
            ΔILD = ${ildGainTxt} · Sem ILD = ${fmtOptNum(mmrMetrics.mmr_semantic_ild, 3)}
          </div>
        </div>
        <div>
          <div class="kpi-label">Phủ nhóm ĐRL (MMR vs P6)</div>
          <div class="kpi-val mono" style="font-size:13.5px">
            ${mmrMetrics.mmr_group_coverage ?? 0} nhóm <span class="text-muted" style="font-size:11.5px">vs ${mmrMetrics.p6_topn_group_coverage ?? 0}</span>
          </div>
          <div class="helper-text mono" style="font-size:11px">
            [${escapeHtml(mmrGroupsTxt)}] vs [${escapeHtml(p6GroupsTxt)}]
          </div>
        </div>
        <div>
          <div class="kpi-label">Cosine cặp E5 (${escapeHtml(simCal.mode_applied || "raw")})</div>
          <div class="kpi-val mono" style="font-size:13.5px">${simRangeTxt}</div>
          <div class="helper-text mono" style="font-size:11px">
            Avg P6: ${fmtOptNum(mmrMetrics.mmr_avg_score, 3)} vs ${fmtOptNum(mmrMetrics.p6_topn_avg_score, 3)}
          </div>
        </div>
      </div>
    `
    : "";

  const compareRowsCount = Math.min(selected.length, ranked.length);
  const compareTableHtml =
    compareRowsCount > 0
      ? `
        <details style="margin-top:12px" open>
          <summary class="helper-text" style="cursor:pointer;font-weight:700">So sánh nhanh Top-${compareRowsCount}: Scoring gốc P6 (chưa MMR) vs Sau MMR (P7)</summary>
          <table class="mmr-compare-table">
            <thead>
              <tr>
                <th style="width:48px">#</th>
                <th>Scoring gốc P6 (chưa MMR)</th>
                <th>Sau chọn lọc MMR (P7)</th>
              </tr>
            </thead>
            <tbody>
              ${Array.from({ length: compareRowsCount }, (_, i) => {
                const rawEv = ranked[i];
                const mmrEv = selected[i];
                const st = mmrEv.selection_trace || {};
                const changed = rawEv.event_id !== mmrEv.event_id;
                const isExp = st.slot_type === "exploration";
                const rowClass = isExp ? "explore-row" : changed ? "changed-row" : "";
                const shiftTxt =
                  Number(st.rank_delta || 0) > 0
                    ? `↑+${st.rank_delta}`
                    : Number(st.rank_delta || 0) < 0
                      ? `↓${st.rank_delta}`
                      : "=";
                return `
                  <tr class="${rowClass}">
                    <td class="mono"><strong>#${i + 1}</strong></td>
                    <td>
                      <span class="badge badge-muted mono">${escapeHtml(rawEv.rubric_item_id || "?")} (${escapeHtml(rawEv.parent_group_id || "?")})</span>
                      <span class="mono text-muted" style="font-size:11px">${fmtOptNum(rawEv.score_total, 3)}</span>
                      <div style="margin-top:3px;font-weight:600">${escapeHtml(rawEv.title)}</div>
                    </td>
                    <td>
                      <span class="badge ${isExp ? "badge-purple" : "badge-teal"} mono">${escapeHtml(isExp ? "KHÁM PHÁ" : st.slot_type === "mandatory" ? "GHIM" : "MMR")} · ${escapeHtml(mmrEv.rubric_item_id || "?")} (${escapeHtml(mmrEv.parent_group_id || "?")})</span>
                      <span class="mono text-muted" style="font-size:11px">${fmtOptNum(mmrEv.score_total, 3)} · ${escapeHtml(shiftTxt)} (P6 #${st.p6_rank ?? i + 1})</span>
                      <div style="margin-top:3px;font-weight:600">${escapeHtml(mmrEv.title)}</div>
                    </td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </details>
      `
      : "";

  const headerHtml = `
    <div class="card">
      <div class="card-title-row">
        <h2 class="card-title">Kết quả Bộ chọn MMR (P7 · Đa dạng hóa Top-N)</h2>
        <span class="mono text-muted" style="font-size:12px">${escapeHtml(res.elapsed_ms)} ms · ${selected.length} / ${sum.eligible_after_p5 ?? 0} ứng viên P6</span>
      </div>
      ${mmrBannerHtml}
      ${compareTableHtml}
      <div class="rec-subtabs" style="margin-top:14px">
        <div class="status-filter-pills">
          <button type="button" class="filter-pill ${state.mmrSubTab === "selected" ? "active" : ""}" data-mmr-subtab="selected">
            Danh sách sau MMR · Top-${selected.length}
          </button>
          <button type="button" class="filter-pill ${state.mmrSubTab === "ranked" ? "active" : ""}" data-mmr-subtab="ranked">
            Scoring gốc P6 · Chưa MMR (${ranked.length})
          </button>
        </div>
        <button type="button" class="btn btn-ghost btn-sm" data-back-scoring="1">← Về trang Scoring (Tab 3)</button>
      </div>
    </div>
  `;

  const isMmrView = state.mmrSubTab === "selected";
  const list = isMmrView ? selected : ranked;
  const bodyHtml = list.length
    ? list.map((ev) => renderRecEventCard(ev, { cal, isMmrView })).join("")
    : `<div class="card empty-state">Không có sự kiện đủ điều kiện P5 để chọn lọc MMR. Hãy điều chỉnh ràng buộc ở Tab 3 (Scoring).</div>`;

  pane.innerHTML = headerHtml + bodyHtml;
  renderMmrProfileSummary();
}

async function runRecommendationQuery() {
  const btn = $("#btnRecRun");
  if (btn) btn.disabled = true;
  try {
    const payload = buildRecPayloadFromForm({ includeMmr: false });
    const res = await apiPost("/api/recommendation/run", payload);
    state.recLastResult = res;
    state.mmrNeedsRefresh = true;
    renderRecommendationResults();
  } catch (err) {
    showToast(`Lỗi chạy Scoring: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function runMmrQuery() {
  const btn = $("#btnMmrRun");
  if (btn) btn.disabled = true;
  try {
    const payload = buildRecPayloadFromForm({ includeMmr: true });
    const res = await apiPost("/api/recommendation/mmr", payload);
    state.mmrLastResult = res;
    state.recLastResult = res;
    state.mmrNeedsRefresh = false;
    renderMmrResults();
    renderRecommendationResults();
  } catch (err) {
    showToast(`Lỗi chạy MMR: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function initRecommendationControls(recBoot) {
  state.recBootstrap = recBoot;

  const stats = recBoot.pool_stats || {};
  const dbBadge = $("#recDbCountBadge");
  if (dbBadge) {
    dbBadge.textContent = `${recBoot.db_path || "drl_database.db"} · ${stats.db_active_events_count ?? 0} sự kiện · ${stats.predicted_count ?? 0} PREDICTED`;
  }
  const embHint = $("#recEmbeddingHint");
  if (embHint && Array.isArray(recBoot.rec_embedding_fields)) {
    embHint.textContent = `Event vector (passage): ${recBoot.rec_embedding_fields.join(" + ")}`;
  }

  // Personas
  $("#recPersonaBar").innerHTML = (recBoot.personas || [])
    .map(
      (p) => `
        <button type="button" class="persona-btn" data-persona-id="${escapeHtml(p.id)}" title="${escapeHtml(p.description || "")}">
          ${escapeHtml(p.label)}
        </button>
      `
    )
    .join("");

  // Interest chips
  $("#recChipsGrid").innerHTML = (recBoot.interest_chips || [])
    .map(
      (c) => `
        <label class="interest-toggle">
          <input type="checkbox" value="${escapeHtml(c.id)}" />
          <span>${escapeHtml(c.label)}</span>
        </label>
      `
    )
    .join("");

  // Reference now quick presets
  $("#recRefNowPresets").innerHTML = (recBoot.reference_now_presets || [])
    .map(
      (p) => `
        <button type="button" class="chip-btn mono" data-ref-now="${escapeHtml(p.value.slice(0, 10))}">
          ${escapeHtml(p.label)}
        </button>
      `
    )
    .join("");

  // Credit presets
  $("#recCreditPreset").innerHTML = (recBoot.credit_presets || [])
    .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
    .join("");

  // Schedule presets
  $("#recSchedulePreset").innerHTML = (recBoot.schedule_presets || [])
    .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
    .join("");

  // Weight presets (+ custom)
  const wEntries = Object.entries(recBoot.weight_presets || {});
  $("#recWeightPreset").innerHTML =
    wEntries
      .map(([k, v]) => `<option value="${escapeHtml(k)}">${escapeHtml(v.label || k)}</option>`)
      .join("") + `<option value="custom">Tùy chỉnh tay (custom)</option>`;

  applyRecPersona("balanced_cs1");
  applyMmrDefaultsToForm(recBoot.mmr_defaults);
}

/* =========================================================================
   EVENT LISTENERS & INIT
   ========================================================================= */

function setupListeners() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = btn.dataset.tab;
      switchTab(target);
      if (target === "mmr" && (state.mmrNeedsRefresh || !state.mmrLastResult)) {
        await runMmrQuery();
      }
    });
  });

  // Clear cached post_id if user edits title, description or format in Tab 1
  ["#clsTitle", "#clsDesc", "#clsFormat"].forEach((sel) => {
    const el = $(sel);
    if (el) {
      el.addEventListener("input", () => {
        $("#clsPostId").value = "";
      });
    }
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
      if ($("#clsFormat")) $("#clsFormat").value = sample.format || "";
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
    if ($("#clsFormat")) {
      $("#clsFormat").value = Array.isArray(ev.format) ? ev.format.join(", ") : ev.format || "";
    }
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
    btn.textContent = "Đang chạy DB...";
    try {
      const res = await apiPost("/api/dual-branch/run", {
        sample_branch: $("#batchSampleVec") ? $("#batchSampleVec").value : undefined,
        rubric_branch: $("#batchRubricVec") ? $("#batchRubricVec").value : undefined,
      });
      state.dualBranchSummary = res.dual_branch_events.summary;
      state.dualBranchEvents = res.dual_branch_events.events;
      renderBatchBrowser();
      if (state.selectedBatchPostId) {
        await selectBatchEvent(state.selectedBatchPostId);
      }
      const recBoot = await apiGet("/api/recommendation/bootstrap");
      state.recBootstrap = recBoot;
      const stats = recBoot.pool_stats || {};
      if ($("#recDbCountBadge")) {
        $("#recDbCountBadge").textContent = `${recBoot.db_path || "drl_database.db"} · ${stats.db_active_events_count ?? 0} sự kiện · ${stats.predicted_count ?? 0} PREDICTED`;
      }
      await runRecommendationQuery();
      showToast(
        `DB Pipeline hoàn tất: ${res.status_counts.PREDICTED} PREDICTED, ${res.status_counts.NEEDS_REVIEW} NEEDS_REVIEW`
      );
    } catch (err) {
      showToast(`Lỗi chạy pipeline: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Chạy lại Pipeline (DB)";
    }
  });

  // --- Tab 3: Scoring (P5-P6) Listeners ---
  $("#recPersonaBar").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-persona-id]");
    if (!btn) return;
    applyRecPersona(btn.dataset.personaId);
    await runRecommendationQuery();
  });

  $("#recChipsGrid").addEventListener("change", (e) => {
    const inp = e.target.closest("input[type='checkbox']");
    if (!inp) return;
    inp.closest(".interest-toggle").classList.toggle("checked", inp.checked);
  });

  $("#recRefNowPresets").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-ref-now]");
    if (!btn) return;
    $("#recRefNow").value = btn.dataset.refNow;
  });

  $("#recCreditPreset").addEventListener("change", (e) => {
    applyCreditPresetToForm(e.target.value);
  });

  ["#recPtsC1", "#recPtsC2", "#recPtsC3", "#recPtsC4", "#recPtsC5", "#recTargetTotal"].forEach(
    (sel) => {
      $(sel).addEventListener("input", updateRecCurrentTotalBadge);
    }
  );

  $("#recSchedulePreset").addEventListener("change", (e) => {
    applySchedulePresetToForm(e.target.value);
  });

  $("#btnRecAddSlot").addEventListener("click", () => {
    const dow = Number($("#recNewSlotDay").value || 0);
    const s = Number($("#recNewSlotStart").value || 13);
    const e = Number($("#recNewSlotEnd").value || 17);
    if (e <= s) {
      showToast("Giờ kết thúc phải lớn hơn giờ bắt đầu.");
      return;
    }
    state.recWeeklySlots.push({
      day_of_week: dow,
      start_hour: s,
      end_hour: e,
      label: "Lịch bận tùy chỉnh",
      location: $("#recHomeCampus").value,
      is_exam: false,
    });
    renderRecSlotsList();
  });

  $("#recSlotsList").addEventListener("click", (e) => {
    const rmSlot = e.target.closest("button[data-remove-slot]");
    if (rmSlot) {
      state.recWeeklySlots.splice(Number(rmSlot.dataset.removeSlot), 1);
      renderRecSlotsList();
      return;
    }
    const rmBusy = e.target.closest("button[data-remove-busy]");
    if (rmBusy) {
      state.recBusyIntervals.splice(Number(rmBusy.dataset.removeBusy), 1);
      renderRecSlotsList();
    }
  });

  $("#recWeightPreset").addEventListener("change", (e) => {
    if (e.target.value !== "custom") {
      applyWeightPresetToForm(e.target.value);
    } else {
      $("#recWeightSummaryBadge").textContent = "custom";
    }
  });

  ["#recW1", "#recW2", "#recW3", "#recW4"].forEach((sel) => {
    $(sel).addEventListener("input", () => {
      $("#recWeightPreset").value = "custom";
      $("#recWeightSummaryBadge").textContent = "custom";
    });
  });

  ["#recMmrTopN", "#recMmrLambda", "#recMmrAlpha"].forEach((sel) => {
    const el = $(sel);
    if (el) {
      el.addEventListener("input", updateRecMmrSummaryBadge);
      el.addEventListener("change", updateRecMmrSummaryBadge);
    }
  });

  $("#recForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await runRecommendationQuery();
  });

  $("#btnRecReset").addEventListener("click", async () => {
    applyRecPersona("balanced_cs1");
    await runRecommendationQuery();
    showToast("Đã đặt lại cấu hình Scoring mặc định");
  });

  // --- Tab 4: Dedicated MMR Selector (P7) Listeners ---
  if ($("#mmrForm")) {
    $("#mmrForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      await runMmrQuery();
    });
  }

  if ($("#btnMmrReset")) {
    $("#btnMmrReset").addEventListener("click", async () => {
      applyMmrDefaultsToForm();
      await runMmrQuery();
      showToast("Đã đặt lại tham số MMR mặc định");
    });
  }

  if ($("#btnBackToScoring")) {
    $("#btnBackToScoring").addEventListener("click", () => {
      switchTab("recommend");
    });
  }

  $("#recUserActionState").addEventListener("click", async (e) => {
    if (!e.target.closest("#btnRecClearActions")) return;
    state.recParticipatedIds = [];
    state.recHiddenIds = [];
    renderRecUserActionState();
    await runRecommendationQuery();
  });

  $("#recResultPane").addEventListener("click", async (e) => {
    if (e.target.closest("button[data-open-mmr]")) {
      switchTab("mmr");
      await runMmrQuery();
      return;
    }
    const subBtn = e.target.closest("button[data-rec-subtab]");
    if (subBtn) {
      state.recSubTab = subBtn.dataset.recSubtab;
      renderRecommendationResults();
      return;
    }
    const dropFilterBtn = e.target.closest("button[data-drop-filter]");
    if (dropFilterBtn) {
      state.recDropReasonFilter = dropFilterBtn.dataset.dropFilter;
      renderRecommendationResults();
      return;
    }
    const joinBtn = e.target.closest("button[data-rec-join]");
    if (joinBtn) {
      const eid = joinBtn.dataset.recJoin;
      if (!state.recParticipatedIds.includes(eid)) state.recParticipatedIds.push(eid);
      renderRecUserActionState();
      await runRecommendationQuery();
      showToast(`Đã đánh dấu tham gia ${eid} → kiểm tra tại tab Bị lọc cứng P5`);
      return;
    }
    const hideBtn = e.target.closest("button[data-rec-hide]");
    if (hideBtn) {
      const eid = hideBtn.dataset.recHide;
      if (!state.recHiddenIds.includes(eid)) state.recHiddenIds.push(eid);
      renderRecUserActionState();
      await runRecommendationQuery();
      showToast(`Đã ẩn ${eid} → P5 xử lý (sự kiện bắt buộc vẫn được giữ lại)`);
    }
  });

  if ($("#mmrResultPane")) {
    $("#mmrResultPane").addEventListener("click", async (e) => {
      if (e.target.closest("button[data-back-scoring]")) {
        switchTab("recommend");
        return;
      }
      const subBtn = e.target.closest("button[data-mmr-subtab]");
      if (subBtn) {
        state.mmrSubTab = subBtn.dataset.mmrSubtab;
        renderMmrResults();
        return;
      }
      const joinBtn = e.target.closest("button[data-rec-join]");
      if (joinBtn) {
        const eid = joinBtn.dataset.recJoin;
        if (!state.recParticipatedIds.includes(eid)) state.recParticipatedIds.push(eid);
        renderRecUserActionState();
        await runMmrQuery();
        showToast(`Đã đánh dấu tham gia ${eid} → đã loại khỏi ứng viên MMR`);
        return;
      }
      const hideBtn = e.target.closest("button[data-rec-hide]");
      if (hideBtn) {
        const eid = hideBtn.dataset.recHide;
        if (!state.recHiddenIds.includes(eid)) state.recHiddenIds.push(eid);
        renderRecUserActionState();
        await runMmrQuery();
        showToast(`Đã ẩn ${eid} → đã cập nhật danh sách MMR`);
      }
    });
  }
}

async function init() {
  setupListeners();
  const [data, recBoot] = await Promise.all([
    apiGet("/api/bootstrap"),
    apiGet("/api/recommendation/bootstrap"),
  ]);
  state.bootstrap = data;
  state.dualBranchSummary = data.dual_branch_summary || null;
  state.dualBranchEvents = (data.dual_branch_events && data.dual_branch_events.events) || [];

  const st = data.embedder_status;
  if (st) {
    $("#modelStatusPill").textContent = `${st.active_model} · ${st.dimension}d`;
  }

  syncDefaultParams();
  renderBatchBrowser();
  initRecommendationControls(recBoot);

  await runLiveClassify({ useCachedQuery: false });
  if (state.dualBranchEvents.length) {
    await selectBatchEvent(state.dualBranchEvents[0].post_id);
  }
  await runRecommendationQuery();
}

init().catch((err) => {
  console.error(err);
  $("#classifyResultPane").innerHTML = `<div class="card empty-state">Không kết nối được backend: ${escapeHtml(err.message)}</div>`;
});
