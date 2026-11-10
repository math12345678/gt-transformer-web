/**
 * Frontend controller for gt-transformer (RDP5 / OpenRDP-style Web Workbench).
 * Plain, dependency-free JS driving the 2x2 desktop panes:
 *   1. Top overview strip + Top-Left FASTA nucleotide alignment viewer
 *   2. Top-Right tabbed event/diagnostic/timing/10k-validation tables
 *   3. Bottom-Left RDP5-style boundary scan plot + method controls
 *   4. Bottom-Right mosaic parental-tract map
 */

(function () {
  "use strict";

  const PALETTE = [
    "#2e8b57", // sea green (RDP5 parent tract green)
    "#3a6ea5", // steel blue
    "#b07830", // ochre brown
    "#b2473e", // brick red
    "#705898", // muted violet
    "#1e8080", // teal
    "#6b7a40", // olive
    "#8c4a6b", // plum
    "#556677", // slate
  ];

  const state = {
    sourceMode: "preset",
    uploadedFastaFile: null,
    uploadedPastedFasta: "",
    uploadedGtFile: null,
    uploadedPastedGt: "",
    presets: [],
    checkpoints: [],
    scoringModes: [],
    currentRun: null,
    selectedEventId: null,
    selectedCarrierSeqIdx: null,
    alnStart: 0,
    alnWidth: 105,
    alnShowCarriersOnly: false,
    lastAlnWindow: null,
    timingHistory: [],
    annotations: [],
    canvasMarkers: [],
  };

  const el = (id) => document.getElementById(id);

  function fmtInt(n) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
    return Number(n).toLocaleString("en-US");
  }

  function fmtFloat(n, digits = 4) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
    return Number(n).toFixed(digits);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function init() {
    bindUIEvents();
    await loadPresetsAndConfig();
    await Promise.all([loadTimingHistory(), loadAnnotations(), loadBenchmarkSummaries()]);
    await submitInference();
  }

  function bindUIEvents() {
    el("presetSelect").addEventListener("change", () => {
      state.sourceMode = "preset";
      submitInference();
    });

    el("btnRunTop").addEventListener("click", submitInference);
    el("btnRunSide").addEventListener("click", submitInference);

    el("scoringModeSelect").addEventListener("change", () => {
      const m = el("scoringModeSelect").value;
      if (m === "global_kl") el("zThreshInput").value = "3.0";
      else if (m === "hybrid_flank" || m === "flank_kl") el("zThreshInput").value = "6.0";
      updateIniPreview();
    });

    // Tab switching
    document.querySelectorAll(".rdp-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
    });

    // Exports
    el("btnExportOpenRdpTsv").addEventListener("click", exportOpenRdpTsv);
    el("btnExportBpCsv").addEventListener("click", exportBreakpointsCsv);
    el("btnExportSeqCsv").addEventListener("click", exportSequenceCallsCsv);
    el("btnExportJson").addEventListener("click", exportFullJson);

    // Open / Upload Modal
    const openUploadModal = () => el("openModal").classList.remove("hidden");
    const closeUploadModal = () => el("openModal").classList.add("hidden");
    el("btnOpenFasta").addEventListener("click", openUploadModal);
    el("btnOpenSide").addEventListener("click", openUploadModal);
    el("btnCloseOpenModal").addEventListener("click", closeUploadModal);
    el("btnCancelOpenModal").addEventListener("click", closeUploadModal);
    el("btnClearUpload").addEventListener("click", () => {
      state.sourceMode = "preset";
      state.uploadedFastaFile = null;
      state.uploadedPastedFasta = "";
      state.uploadedGtFile = null;
      state.uploadedPastedGt = "";
      el("fastaFileInput").value = "";
      el("pastedFastaInput").value = "";
      el("gtFileInput").value = "";
      el("pastedGtInput").value = "";
      closeUploadModal();
      submitInference();
    });
    el("uploadForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      state.uploadedFastaFile = el("fastaFileInput").files[0] || null;
      state.uploadedPastedFasta = el("pastedFastaInput").value.trim();
      state.uploadedGtFile = el("gtFileInput").files[0] || null;
      state.uploadedPastedGt = el("pastedGtInput").value.trim();
      if (!state.uploadedFastaFile && !state.uploadedPastedFasta) {
        alert("Choose a FASTA file or paste FASTA sequence text.");
        return;
      }
      state.sourceMode = "upload";
      closeUploadModal();
      await submitInference();
    });

    // Options Modal
    const openOpts = () => {
      updateIniPreview();
      el("optionsModal").classList.remove("hidden");
    };
    const closeOpts = () => el("optionsModal").classList.add("hidden");
    el("btnOptionsTop").addEventListener("click", openOpts);
    el("btnOptionsSide").addEventListener("click", openOpts);
    el("btnCloseOptionsModal").addEventListener("click", closeOpts);
    el("btnCancelOptions").addEventListener("click", closeOpts);
    el("btnApplyOptions").addEventListener("click", async () => {
      closeOpts();
      await submitInference();
    });
    ["zThreshInput", "tfLogitThreshInput", "flankWindowInput", "mergeRadiusInput", "betaInput", "nullPolicyInput"].forEach(
      (id) => el(id).addEventListener("input", updateIniPreview)
    );

    // Alignment Viewer navigation
    el("btnAlnPrevFast").addEventListener("click", () => moveAlignmentWindow(-500));
    el("btnAlnPrev").addEventListener("click", () => moveAlignmentWindow(-80));
    el("btnAlnNext").addEventListener("click", () => moveAlignmentWindow(80));
    el("btnAlnNextFast").addEventListener("click", () => moveAlignmentWindow(500));
    el("btnAlnGo").addEventListener("click", () => {
      const val = parseInt(el("alnJumpInput").value, 10) || 0;
      loadAlignmentWindow(val);
    });
    el("btnAlnToEvent").addEventListener("click", jumpAlignmentToSelectedEvent);
    el("btnToggleCarriersOnly").addEventListener("click", () => {
      state.alnShowCarriersOnly = !state.alnShowCarriersOnly;
      el("btnToggleCarriersOnly").textContent = state.alnShowCarriersOnly
        ? "Show All Sequences"
        : "Carriers + Parents Only";
      if (state.lastAlnWindow) renderAlignmentWindow(state.lastAlnWindow);
    });

    // Mosaic filter
    el("chkMosaicSelectedOnly").addEventListener("change", () => {
      if (state.currentRun) drawMosaicMap(state.currentRun);
    });

    // Annotation modal
    el("btnOpenAnnotationModal").addEventListener("click", () => openAnnotationModal("general", ""));
    el("btnCloseModal").addEventListener("click", closeAnnotationModal);
    el("btnCancelModal").addEventListener("click", closeAnnotationModal);
    el("annotationForm").addEventListener("submit", handleSaveAnnotation);

    // Canvas interactions
    const plotCanvas = el("genomeTrackCanvas");
    plotCanvas.addEventListener("mousemove", handleCanvasHover);
    plotCanvas.addEventListener("mouseleave", () => el("trackTooltip").classList.add("hidden"));
    plotCanvas.addEventListener("click", handleCanvasClick);

    el("overviewStripCanvas").addEventListener("click", (e) => {
      if (!state.currentRun) return;
      const rect = e.target.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
      const bp = Math.round(frac * state.currentRun.alignment.genome_length_bp);
      loadAlignmentWindow(Math.max(0, bp - Math.floor(state.alnWidth / 2)));
    });

    window.addEventListener("resize", () => {
      if (state.currentRun) {
        drawOverviewStrip(state.currentRun);
        drawGenomeTrack(state.currentRun);
        drawMosaicMap(state.currentRun);
      }
    });
  }

  function switchTab(tabId) {
    document.querySelectorAll(".rdp-tab").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-content").forEach((p) => {
      p.classList.toggle("hidden", p.id !== tabId);
    });
  }

  function updateIniPreview() {
    const ini = [
      `[GTTransformer]`,
      `checkpoint = ${el("checkpointSelect").value || "ckpt_N10000_extreme"}`,
      `scoring_mode = ${el("scoringModeSelect").value || "hybrid_flank"}`,
      `null_policy = ${el("nullPolicyInput").value}`,
      `tf_logit_thresh = ${el("tfLogitThreshInput").value}`,
      ``,
      `[FlankKL]`,
      `flank_window = ${el("flankWindowInput").value}`,
      `z_thresh = ${el("zThreshInput").value}`,
      `beta = ${el("betaInput").value}`,
      `merge_radius_bp = ${el("mergeRadiusInput").value}`,
      `eval_tolerance_bp = ${el("evalToleranceInput").value}`,
    ].join("\n");
    el("iniPreviewBox").textContent = ini;
  }

  async function loadPresetsAndConfig() {
    const res = await fetch("/api/presets");
    const data = await res.json();
    state.presets = data.presets || [];
    state.checkpoints = data.checkpoints || [];
    state.scoringModes = data.scoring_modes || [];

    el("presetSelect").innerHTML = state.presets
      .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`)
      .join("");
    el("checkpointSelect").innerHTML = state.checkpoints
      .map((c) => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`)
      .join("");
    el("scoringModeSelect").innerHTML = state.scoringModes
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
      .join("");
    updateIniPreview();
  }

  async function submitInference() {
    const btnTop = el("btnRunTop");
    const btnSide = el("btnRunSide");
    btnTop.disabled = true;
    btnSide.disabled = true;
    btnTop.textContent = "Running...";
    btnSide.textContent = "Running...";

    try {
      const fd = new FormData();
      fd.append("checkpoint", el("checkpointSelect").value);
      fd.append("scoring_mode", el("scoringModeSelect").value);
      fd.append("null_policy", el("nullPolicyInput").value);
      fd.append("flank_window", el("flankWindowInput").value);
      fd.append("beta", el("betaInput").value);
      fd.append("z_thresh", el("zThreshInput").value);
      fd.append("tf_logit_thresh", el("tfLogitThreshInput").value);
      fd.append("merge_radius_bp", el("mergeRadiusInput").value);
      fd.append("eval_tolerance_bp", el("evalToleranceInput").value);

      if (state.sourceMode === "preset") {
        fd.append("preset_id", el("presetSelect").value);
      } else {
        if (state.uploadedFastaFile) {
          fd.append("fasta_file", state.uploadedFastaFile);
        } else if (state.uploadedPastedFasta) {
          fd.append("pasted_fasta", state.uploadedPastedFasta);
        }
        if (state.uploadedGtFile) {
          fd.append("gt_file", state.uploadedGtFile);
        } else if (state.uploadedPastedGt) {
          fd.append("pasted_gt", state.uploadedPastedGt);
        }
      }

      const res = await fetch("/api/infer", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Inference failed" }));
        alert("Error: " + (err.detail || res.statusText));
        return;
      }

      const data = await res.json();
      state.currentRun = data;
      const evs = data.consolidated_events || [];
      state.selectedEventId = evs.length > 0 ? evs[0].event_id : null;
      state.selectedCarrierSeqIdx =
        evs.length > 0 && evs[0].recombinants && evs[0].recombinants.length > 0
          ? evs[0].recombinants[0].seq_idx
          : null;

      renderRun(data);
      const startBp =
        evs.length > 0 ? Math.max(0, evs[0].most_plausible_bp_orig - Math.floor(state.alnWidth / 2)) : 0;
      await Promise.all([loadAlignmentWindow(startBp), loadTimingHistory()]);
    } finally {
      btnTop.disabled = false;
      btnSide.disabled = false;
      btnTop.textContent = "Run";
      btnSide.textContent = "Run";
    }
  }

  function renderRun(run) {
    const aln = run.alignment;
    const tim = run.timings;
    const diag = run.diagnostics;
    const evs = run.consolidated_events || [];
    const rawPeaks = run.raw_peaks || [];

    el("engineStatusBadge").textContent =
      `live · ${fmtInt(run.model_param_count)} params · ${tim.run_id}`;

    el("sumDimensions").textContent = `${aln.n_sequences} × ${fmtInt(aln.genome_length_bp)} bp`;
    el("sumCols").textContent = `${fmtInt(aln.n_informative_cols)}→${fmtInt(aln.n_transformed_cols)} (W=${aln.window_size})`;
    el("sumEvents").textContent = `${evs.length} cons / ${rawPeaks.length} raw`;

    if (diag.has_ground_truth && diag.tolerance_metrics.length > 0) {
      const m =
        diag.tolerance_metrics.find((x) => x.tolerance_bp === diag.active_tolerance_bp) ||
        diag.tolerance_metrics[diag.tolerance_metrics.length - 1];
      if (diag.n_ground_truth_breakpoints === 0) {
        el("sumRecall").textContent = "N/A (0 GT)";
        el("sumPrecision").textContent = m.fp === 0 ? "0 FP (100% clean)" : `${m.fp} FP`;
        el("sumMedErr").textContent = "—";
      } else {
        el("sumRecall").textContent = `${(m.recall * 100).toFixed(1)}% (${m.tp}/${diag.n_ground_truth_breakpoints})`;
        el("sumPrecision").textContent = `P=${(m.precision * 100).toFixed(1)}% F1=${m.f1.toFixed(3)}`;
        el("sumMedErr").textContent = m.median_error_bp !== null ? `${m.median_error_bp.toFixed(1)} bp` : "—";
      }
    } else {
      el("sumRecall").textContent = "Unlabelled";
      el("sumPrecision").textContent = "Unlabelled";
      el("sumMedErr").textContent = "—";
    }
    el("sumTime").textContent = `${tim.total_ms.toFixed(0)} ms (${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s)`;

    el("presetMetaLine").textContent =
      `${aln.dataset_name} (${aln.source_type}) | mode=${run.config.scoring_mode} | ckpt=${run.config.checkpoint} | GC=${aln.gc_content_pct}%`;

    drawOverviewStrip(run);
    drawGenomeTrack(run);
    drawMosaicMap(run);
    renderOverviewAndParents(run);
    renderDiagnosticsTab(run);
    renderCurrentTimings(tim);
  }

  // -------------------------------------------------------------------------
  // Top-Left: Real FASTA Alignment Viewer
  // -------------------------------------------------------------------------
  async function loadAlignmentWindow(startBp) {
    if (!state.currentRun) return;
    const runId = state.currentRun.timings.run_id;
    const L = state.currentRun.alignment.genome_length_bp;
    const clampedStart = Math.max(0, Math.min(Math.max(0, L - state.alnWidth), Math.round(startBp)));
    state.alnStart = clampedStart;
    el("alnJumpInput").value = String(clampedStart);

    const res = await fetch(
      `/api/alignment/window?run_id=${encodeURIComponent(runId)}&start=${clampedStart}&width=${state.alnWidth}`
    );
    if (!res.ok) return;
    const data = await res.json();
    state.lastAlnWindow = data;
    renderAlignmentWindow(data);
  }

  function moveAlignmentWindow(delta) {
    loadAlignmentWindow(state.alnStart + delta);
  }

  function jumpAlignmentToSelectedEvent() {
    if (!state.currentRun) return;
    const ev = (state.currentRun.consolidated_events || []).find((x) => x.event_id === state.selectedEventId);
    const bp = ev ? ev.most_plausible_bp_orig : 0;
    loadAlignmentWindow(Math.max(0, bp - Math.floor(state.alnWidth / 2)));
  }

  function renderAlignmentWindow(win) {
    const start = win.start_bp;
    const end = win.end_bp;
    const infoSet = new Set(win.informative_cols || []);
    el("alnPaneTitle").textContent =
      `Sequence Alignment (${fmtInt(start + 1)}..${fmtInt(end)} of ${fmtInt(win.genome_length_bp)} bp)`;

    // Identify carriers & parents of selected event for highlighting
    const selectedEv = state.currentRun
      ? (state.currentRun.consolidated_events || []).find((x) => x.event_id === state.selectedEventId)
      : null;
    const carrierSet = new Set(selectedEv ? selectedEv.recombinant_sequences : []);
    const parentSet = new Set();
    if (selectedEv && selectedEv.recombinants) {
      selectedEv.recombinants.forEach((r) => {
        parentSet.add(r.ext_parent_left);
        parentSet.add(r.ext_parent_right);
      });
    }
    const evBp = selectedEv ? selectedEv.most_plausible_bp_orig : -1;

    let seqRows = win.sequences || [];
    if (state.alnShowCarriersOnly && (carrierSet.size > 0 || parentSet.size > 0)) {
      seqRows = seqRows.filter((s) => carrierSet.has(s.seq_idx) || parentSet.has(s.seq_idx));
    }

    // Build ruler header
    let rulerCells = `<th style="padding:1px 5px;">Seq</th><th>`;
    for (let pos = start; pos < end; pos++) {
      const pos1 = pos + 1;
      const mark = pos1 % 10 === 0 ? "|" : pos1 % 5 === 0 ? ":" : ".";
      rulerCells += `<span class="nt-cell" title="pos ${pos1}">${mark}</span>`;
    }
    rulerCells += `</th>`;

    // Build nucleotide rows
    const bodyHtml = seqRows
      .map((row) => {
        const isC = carrierSet.has(row.seq_idx);
        const isP = !isC && parentSet.has(row.seq_idx);
        const cls = isC ? "aln-seq-row aln-row-carrier" : isP ? "aln-seq-row aln-row-parent" : "aln-seq-row";
        let seqHtml = "";
        const txt = row.slice || "";
        for (let k = 0; k < txt.length; k++) {
          const absPos = start + k;
          const clsList = ["nt-cell"];
          if (infoSet.has(absPos)) clsList.push("nt-info");
          if (absPos === evBp) clsList.push("nt-bp");
          seqHtml += `<span class="${clsList.join(" ")}">${escapeHtml(txt[k])}</span>`;
        }
        return `<tr class="${cls}"><td class="aln-row-idx">${row.seq_idx}</td><td>${seqHtml}</td></tr>`;
      })
      .join("");

    el("alnMatrixContainer").innerHTML =
      `<table class="aln-matrix-table"><thead><tr class="aln-ruler-row">${rulerCells}</tr></thead><tbody>${bodyHtml}</tbody></table>`;

    // Sidebar sequence list (like RDP5 "Show All Sequences")
    el("alnSeqList").innerHTML = (win.sequences || [])
      .map((row) => {
        const isC = carrierSet.has(row.seq_idx);
        const isP = !isC && parentSet.has(row.seq_idx);
        const cls = isC ? "seq-list-item is-carrier" : isP ? "seq-list-item is-parent" : "seq-list-item";
        const tag = isC ? " [REC]" : isP ? " [PAR]" : "";
        return `<div class="${cls}" title="${escapeHtml(row.name)}">${row.seq_idx}: ${escapeHtml(row.name)}${tag}</div>`;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Top-Right: Overview, Parents, Diagnostics, Timings, 10k Validation
  // -------------------------------------------------------------------------
  function statusTagHtml(status) {
    if (status === "TP") return `<span class="tag tag-tp">TP</span>`;
    if (status === "FP") return `<span class="tag tag-fp">FP</span>`;
    if (status === "FN") return `<span class="tag tag-fn">FN</span>`;
    return `<span class="tag tag-unlabelled">UNLABELLED</span>`;
  }

  function selectEvent(eventId, jumpAlign = false) {
    if (!state.currentRun) return;
    state.selectedEventId = eventId;
    const ev = (state.currentRun.consolidated_events || []).find((x) => x.event_id === eventId);
    if (ev && ev.recombinants && ev.recombinants.length > 0) {
      state.selectedCarrierSeqIdx = ev.recombinants[0].seq_idx;
    } else {
      state.selectedCarrierSeqIdx = null;
    }
    renderOverviewAndParents(state.currentRun);
    drawGenomeTrack(state.currentRun);
    drawMosaicMap(state.currentRun);
    if (jumpAlign) {
      jumpAlignmentToSelectedEvent();
    } else if (state.lastAlnWindow) {
      renderAlignmentWindow(state.lastAlnWindow);
    }
  }

  function renderOverviewAndParents(run) {
    const evs = run.consolidated_events || [];
    const l1Body = el("level1EventsBody");

    if (evs.length === 0) {
      l1Body.innerHTML = `<tr><td colspan="10">No consolidated recombination events detected above threshold.</td></tr>`;
      el("level2CarriersBody").innerHTML = `<tr><td colspan="7">No event selected.</td></tr>`;
      el("level3ParentsContainer").innerHTML = `<div class="mono-muted">No carrier selected.</div>`;
      return;
    }

    l1Body.innerHTML = evs
      .map((ev) => {
        const sel = ev.event_id === state.selectedEventId ? ' class="selected-row"' : "";
        const lead = ev.recombinants && ev.recombinants[0] ? ev.recombinants[0] : null;
        const parentsStr = lead ? `${lead.ext_parent_left} → ${lead.ext_parent_right}` : "—";
        return (
          `<tr${sel} data-row-event="${ev.event_id}">` +
          `<td class="num"><strong>${ev.event_id}</strong></td>` +
          `<td class="mono">${escapeHtml(run.config.scoring_mode)}</td>` +
          `<td class="num"><strong>${fmtInt(ev.most_plausible_bp_orig)}</strong></td>` +
          `<td class="mono">${fmtInt(ev.span_orig_start)}..${fmtInt(ev.span_orig_end)}</td>` +
          `<td class="num">${ev.n_peaks_consolidated}</td>` +
          `<td class="num">${ev.score_z.toFixed(2)}</td>` +
          `<td class="mono">${ev.tf_logit.toFixed(2)} (${ev.tf_prob.toFixed(2)})</td>` +
          `<td class="mono">${escapeHtml(ev.recombinant_sequences.slice(0, 6).join(","))}</td>` +
          `<td class="mono">${escapeHtml(parentsStr)}</td>` +
          `<td>${statusTagHtml(ev.match_status)}</td>` +
          `</tr>`
        );
      })
      .join("");

    l1Body.querySelectorAll("tr[data-row-event]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectEvent(Number(tr.getAttribute("data-row-event")), true);
      });
    });

    const selectedEv = evs.find((x) => x.event_id === state.selectedEventId) || evs[0];
    el("level2HeaderTitle").textContent =
      `Level 2 Recombinant Sequences — Event #${selectedEv.event_id} (${fmtInt(selectedEv.most_plausible_bp_orig)} bp)`;

    const recombs = selectedEv.recombinants || [];
    const l2Body = el("level2CarriersBody");
    if (recombs.length === 0) {
      l2Body.innerHTML = `<tr><td colspan="7">No carrier sequences recorded.</td></tr>`;
      el("level3ParentsContainer").innerHTML = "";
      return;
    }

    l2Body.innerHTML = recombs
      .map((r) => {
        const sel = r.seq_idx === state.selectedCarrierSeqIdx ? ' class="selected-row"' : "";
        return (
          `<tr${sel} data-row-carrier="${r.seq_idx}">` +
          `<td class="num"><strong>${r.seq_idx}</strong></td>` +
          `<td class="mono">${escapeHtml(r.seq_name)}</td>` +
          `<td class="num">${fmtFloat(r.p_switch, 4)}</td>` +
          `<td class="mono">${r.parent_left} (ext: ${r.ext_parent_left})</td>` +
          `<td class="mono">${r.parent_right} (ext: ${r.ext_parent_right})</td>` +
          `<td class="num">${fmtFloat(r.left_sim, 4)}</td>` +
          `<td class="num">${fmtFloat(r.right_sim, 4)}</td>` +
          `</tr>`
        );
      })
      .join("");

    l2Body.querySelectorAll("tr[data-row-carrier]").forEach((tr) => {
      tr.addEventListener("click", () => {
        state.selectedCarrierSeqIdx = Number(tr.getAttribute("data-row-carrier"));
        renderOverviewAndParents(state.currentRun);
      });
    });

    const selectedCarrier =
      recombs.find((r) => r.seq_idx === state.selectedCarrierSeqIdx) || recombs[0];
    renderLevel3Parents(selectedEv, selectedCarrier);
  }

  function renderLevel3Parents(ev, carrier) {
    el("level3HeaderTitle").textContent =
      `Level 3 Ranked Parental Candidates — Carrier Seq ${carrier.seq_idx} (${carrier.seq_name})`;
    const mkTable = (title, cands) => {
      const rows = (cands || [])
        .map(
          (c, i) =>
            `<tr>` +
            `<td class="num">#${i + 1}</td>` +
            `<td class="num"><strong>${c.seq_idx}</strong></td>` +
            `<td class="mono">${escapeHtml(c.seq_name)}</td>` +
            `<td class="num">${fmtFloat(c.similarity, 4)}</td>` +
            `<td class="mono">${c.is_external_to_event ? "external" : "co-carrier"}</td>` +
            `</tr>`
        )
        .join("");
      return (
        `<div class="pane-subhead mt-6">${escapeHtml(title)}</div>` +
        `<div class="table-scroll"><table class="rdp-table">` +
        `<thead><tr><th>Rank</th><th>seq_idx</th><th>Name</th><th>Similarity</th><th>Scope</th></tr></thead>` +
        `<tbody>${rows || '<tr><td colspan="5">None</td></tr>'}</tbody></table></div>`
      );
    };
    el("level3ParentsContainer").innerHTML =
      mkTable(`Left Flank (< ${fmtInt(ev.most_plausible_bp_orig)} bp)`, carrier.left_candidates) +
      mkTable(`Right Flank (> ${fmtInt(ev.most_plausible_bp_orig)} bp)`, carrier.right_candidates);
  }

  function renderDiagnosticsTab(run) {
    const diag = run.diagnostics;
    const evs = run.consolidated_events || [];

    el("diagVerdictBanner").textContent = (diag.summary_notes || []).join(" ");

    const tolBody = el("toleranceMetricsBody");
    if (!diag.has_ground_truth || diag.tolerance_metrics.length === 0) {
      tolBody.innerHTML = `<tr><td colspan="10">No ground-truth log loaded. Use Open FASTA... to attach a SantaSim .txt, .json, or .csv ground-truth log.</td></tr>`;
    } else {
      tolBody.innerHTML = diag.tolerance_metrics
        .map((m) => {
          const isAct = m.tolerance_bp === diag.active_tolerance_bp;
          const style = isAct ? ' style="background-color:#e6eef8;font-weight:700;"' : "";
          const sign = m.lift >= 0 ? "+" : "";
          return (
            `<tr${style}>` +
            `<td class="mono">±${m.tolerance_bp} bp</td>` +
            `<td class="num">${m.tp}</td>` +
            `<td class="num">${m.fp}</td>` +
            `<td class="num">${m.fn}</td>` +
            `<td class="num">${(m.recall * 100).toFixed(1)}%</td>` +
            `<td class="num">${(m.precision * 100).toFixed(1)}%</td>` +
            `<td class="num">${m.f1.toFixed(3)}</td>` +
            `<td class="num">${m.control_f1.toFixed(3)}</td>` +
            `<td class="num">${sign}${m.lift.toFixed(3)}</td>` +
            `<td class="num">${m.median_error_bp !== null ? m.median_error_bp.toFixed(1) + " bp" : "—"}</td>` +
            `</tr>`
          );
        })
        .join("");
    }

    const gtBody = el("gtDiagnosticBody");
    if (!diag.has_ground_truth) {
      gtBody.innerHTML = `<tr><td colspan="9">No ground-truth breakpoints loaded.</td></tr>`;
    } else if (diag.ground_truth_items.length === 0) {
      gtBody.innerHTML = `<tr><td colspan="9">Clonal negative control (0 true breakpoints). Any calls below are false positives.</td></tr>`;
    } else {
      gtBody.innerHTML = diag.ground_truth_items
        .map((g) => {
          const st = g.detected ? statusTagHtml("TP") : statusTagHtml("FN");
          const matchedStr = g.detected ? `#${g.matched_pred_event_id} (${fmtInt(g.matched_pred_bp_orig)})` : "—";
          const errStr = g.detected && g.bp_error !== null ? `${g.bp_error}` : "—";
          const localZ =
            g.nearest_subthreshold_score_z !== null
              ? `${g.nearest_subthreshold_score_z.toFixed(2)} @ ${fmtInt(g.nearest_subthreshold_peak_bp)}`
              : "—";
          const targetId = `GT ${g.gt_bp_orig} bp (Event ${g.gt_event_id ?? "?"})`;
          return (
            `<tr>` +
            `<td>${st}</td>` +
            `<td class="num"><strong>${fmtInt(g.gt_bp_orig)}</strong></td>` +
            `<td class="mono">${escapeHtml(g.gt_event_id ?? "—")}</td>` +
            `<td>${escapeHtml(g.event_type)}</td>` +
            `<td class="mono">${matchedStr}</td>` +
            `<td class="num">${errStr}</td>` +
            `<td class="mono">${localZ}</td>` +
            `<td>${escapeHtml(g.failure_hypothesis)}</td>` +
            `<td><button type="button" class="btn-link" data-ann-type="${g.detected ? "predicted_event" : "missed_gt_breakpoint"}" data-ann-id="${escapeHtml(targetId)}">Note</button></td>` +
            `</tr>`
          );
        })
        .join("");
    }

    const predBody = el("predDiagnosticBody");
    if (evs.length === 0) {
      predBody.innerHTML = `<tr><td colspan="8">No predicted calls above threshold.</td></tr>`;
    } else {
      predBody.innerHTML = evs
        .map((ev) => {
          const st = statusTagHtml(ev.match_status);
          const gtStr = ev.matched_gt_bp !== null ? `${fmtInt(ev.matched_gt_bp)}` : "—";
          const errStr = ev.bp_error !== null ? `${ev.bp_error}` : "—";
          const targetId = `Event #${ev.event_id} (${ev.most_plausible_bp_orig} bp)`;
          return (
            `<tr>` +
            `<td>${st}</td>` +
            `<td class="num">#${ev.event_id}</td>` +
            `<td class="num"><strong>${fmtInt(ev.most_plausible_bp_orig)}</strong></td>` +
            `<td class="mono">${ev.score_z.toFixed(2)} / ${ev.tf_logit.toFixed(2)}</td>` +
            `<td class="mono">${gtStr}</td>` +
            `<td class="num">${errStr}</td>` +
            `<td>${escapeHtml(ev.diagnostic_explanation)}</td>` +
            `<td><button type="button" class="btn-link" data-ann-type="predicted_event" data-ann-id="${escapeHtml(targetId)}">Note</button></td>` +
            `</tr>`
          );
        })
        .join("");
    }

    document.querySelectorAll("[data-ann-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        openAnnotationModal(btn.getAttribute("data-ann-type"), btn.getAttribute("data-ann-id"));
      });
    });
  }

  function renderCurrentTimings(tim) {
    el("timingRunSubtitle").textContent =
      `run_id: ${tim.run_id} | total: ${tim.total_ms.toFixed(2)} ms | ${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s`;
    const rowsHtml = (tim.stages || [])
      .map((s) => {
        const itemsStr = s.items_processed !== null ? `${fmtInt(s.items_processed)} ${s.items_unit || ""}` : "—";
        const tpStr = s.throughput !== null ? `${fmtInt(Math.round(s.throughput))} ${s.throughput_unit || ""}` : "—";
        return (
          `<tr>` +
          `<td><strong>${escapeHtml(s.stage_name)}</strong></td>` +
          `<td class="num">${s.duration_ms.toFixed(2)}</td>` +
          `<td class="num">${s.pct_of_total.toFixed(1)}%</td>` +
          `<td class="mono">${escapeHtml(itemsStr)}</td>` +
          `<td class="mono">${escapeHtml(tpStr)}</td>` +
          `<td>${escapeHtml(s.notes)}</td>` +
          `</tr>`
        );
      })
      .join("");
    el("currentTimingBody").innerHTML =
      rowsHtml +
      `<tr style="background-color:#ece9d8;font-weight:700;">` +
      `<td>Total End-to-End</td><td class="num">${tim.total_ms.toFixed(2)}</td><td class="num">100.0%</td>` +
      `<td class="mono">${tim.n_sequences} × ${fmtInt(tim.genome_length_bp)} bp</td>` +
      `<td class="mono">${fmtInt(Math.round(tim.throughput_bp_per_sec))} bp/s</td>` +
      `<td>W=${tim.window_size}, M_tx=${fmtInt(tim.n_transformed_cols)}</td></tr>`;
  }

  async function loadTimingHistory() {
    const res = await fetch("/api/timings");
    const data = await res.json();
    state.timingHistory = data.records || [];
    const tbody = el("historyTimingBody");
    if (state.timingHistory.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12">No live timing runs recorded yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = state.timingHistory
      .map((r) => {
        const byId = {};
        (r.stages || []).forEach((s) => {
          byId[s.stage_id] = s.duration_ms;
        });
        return (
          `<tr>` +
          `<td class="mono">${escapeHtml(r.run_id)}</td>` +
          `<td class="mono">${escapeHtml(r.dataset_name)}</td>` +
          `<td class="mono">${escapeHtml(r.checkpoint)}</td>` +
          `<td class="mono">${escapeHtml(r.scoring_mode)}</td>` +
          `<td class="mono">${r.n_sequences}×${fmtInt(r.genome_length_bp)}</td>` +
          `<td class="num">${fmtInt(r.n_transformed_cols)}</td>` +
          `<td class="num">${(byId.parse_filter ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.pairwise_embed ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.transformer_fwd ?? 0).toFixed(1)}</td>` +
          `<td class="num">${(byId.flank_kl_null ?? 0).toFixed(1)}</td>` +
          `<td class="num"><strong>${r.total_ms.toFixed(1)}</strong></td>` +
          `<td class="num">${fmtInt(Math.round(r.throughput_bp_per_sec))}</td>` +
          `</tr>`
        );
      })
      .join("");
  }

  async function loadBenchmarkSummaries() {
    try {
      const res = await fetch("/api/benchmarks");
      const data = await res.json();
      renderSummaryCsvTable("scalingBenchmarkHead", "scalingBenchmarkBody", data.scaling_table);
      renderSummaryCsvTable("extremeBenchmarkHead", "extremeBenchmarkBody", data.out_of_range_table);
    } catch (err) {
      console.error("Failed to load benchmarks:", err);
    }
  }

  function renderSummaryCsvTable(headId, bodyId, tableObj) {
    const thead = el(headId);
    const tbody = el(bodyId);
    if (!tableObj || !tableObj.headers || tableObj.headers.length === 0) return;
    thead.innerHTML = `<tr>${tableObj.headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
    tbody.innerHTML = tableObj.rows
      .map((row) => {
        if (row[0] && row[0].startsWith("===")) {
          return `<tr style="background-color:#ece9d8;font-weight:700;"><td colspan="${tableObj.headers.length}">${escapeHtml(row[0])}</td></tr>`;
        }
        return `<tr>${row.map((cell, i) => `<td${i > 0 ? ' class="mono"' : ""}>${escapeHtml(cell)}</td>`).join("")}</tr>`;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Canvas 1: Top Genomic Overview Strip (RDP5 Magenta Density Bar)
  // -------------------------------------------------------------------------
  function drawOverviewStrip(run) {
    const canvas = el("overviewStripCanvas");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(500, Math.floor(rect.width));
    const h = 28;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#ffe6ff";
    ctx.fillRect(0, 0, w, h);

    const prof = run.track_profile;
    const L = Math.max(1, run.alignment.genome_length_bp);
    const dens = prof && prof.informative_density ? prof.informative_density : [];
    const pos = prof && prof.positions_orig_bp ? prof.positions_orig_bp : [];

    for (let i = 0; i < dens.length; i++) {
      const x = (pos[i] / L) * w;
      const bw = Math.max(2, w / Math.max(1, dens.length));
      const d = Math.min(1, Math.max(0, dens[i]));
      const alpha = 0.15 + 0.8 * d;
      ctx.fillStyle = `rgba(180, 0, 140, ${alpha.toFixed(2)})`;
      ctx.fillRect(x, 0, bw, h - 8);
    }

    // Bottom horizontal bracket bar for predicted events (like RDP5's bottom track arrows)
    ctx.fillStyle = "#f4f3ee";
    ctx.fillRect(0, h - 9, w, 9);
    ctx.strokeStyle = "#808080";
    ctx.beginPath();
    ctx.moveTo(0, h - 9);
    ctx.lineTo(w, h - 9);
    ctx.stroke();

    (run.consolidated_events || []).forEach((ev) => {
      const x1 = (ev.span_orig_start / L) * w;
      const x2 = Math.max(x1 + 3, (ev.span_orig_end / L) * w);
      ctx.fillStyle = ev.event_id === state.selectedEventId ? "#000000" : "#1a4b8c";
      ctx.fillRect(x1, h - 7, x2 - x1, 5);
    });
  }

  // -------------------------------------------------------------------------
  // Canvas 2: Bottom-Left RDP5 Breakpoint Scan Plot
  // -------------------------------------------------------------------------
  function drawGenomeTrack(run) {
    const canvas = el("genomeTrackCanvas");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(460, Math.floor(rect.width));
    const height = Math.max(210, Math.floor(rect.height));

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const L = Math.max(1, run.alignment.genome_length_bp);
    const padL = 46;
    const padR = 14;
    const topStrip = 24;
    const plotTop = 34;
    const plotBot = height - 26;
    const plotW = width - padL - padR;
    const plotH = plotBot - plotTop;
    const xOfBp = (bp) => padL + (Math.max(0, Math.min(L, bp)) / L) * plotW;

    // Plot frame & faint grid
    ctx.fillStyle = "#fbfbfb";
    ctx.fillRect(padL, plotTop, plotW, plotH);
    ctx.strokeStyle = "#222222";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, plotTop, plotW, plotH);

    // Top GT / Pred tick lane (matching RDP5's top horizontal bar & vertical tick strip)
    ctx.fillStyle = "#111111";
    ctx.font = "bold 10px Consolas, monospace";
    ctx.textAlign = "right";
    ctx.fillText("GT", padL - 5, 11);
    ctx.fillText("PRED", padL - 5, 22);

    state.canvasMarkers = [];
    const gtItems = (run.diagnostics && run.diagnostics.ground_truth_items) || [];
    gtItems.forEach((g) => {
      const gx = xOfBp(g.gt_bp_orig);
      ctx.strokeStyle = g.detected ? "#1b7837" : "#b2182b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(gx, 3);
      ctx.lineTo(gx, 13);
      ctx.stroke();
      state.canvasMarkers.push({
        kind: "gt",
        x: gx,
        y: 8,
        bp: g.gt_bp_orig,
        label: `GT ${fmtInt(g.gt_bp_orig)} bp (${g.detected ? "TP" : "FN"})`,
        detail: g.failure_hypothesis,
      });
    });

    const evs = run.consolidated_events || [];
    evs.forEach((ev) => {
      const px = xOfBp(ev.most_plausible_bp_orig);
      const isTp = ev.match_status === "TP";
      const isSel = ev.event_id === state.selectedEventId;
      ctx.fillStyle = isTp ? "#2166ac" : ev.match_status === "FP" ? "#d6604d" : "#222222";
      ctx.fillRect(px - 2, 14, 4, 10);
      if (isSel) {
        ctx.strokeStyle = "#000000";
        ctx.strokeRect(px - 4, 13, 8, 12);
      }
      state.canvasMarkers.push({
        kind: "pred",
        eventId: ev.event_id,
        x: px,
        y: 19,
        bp: ev.most_plausible_bp_orig,
        label: `Event #${ev.event_id} @ ${fmtInt(ev.most_plausible_bp_orig)} bp (${ev.match_status})`,
        detail: `z=${ev.score_z.toFixed(2)}, TF=${ev.tf_logit.toFixed(2)}`,
      });
    });

    // Curves
    const prof = run.track_profile;
    if (prof && prof.positions_orig_bp && prof.positions_orig_bp.length > 0) {
      const maxZ = Math.max(10.0, prof.threshold_z * 1.35, ...prof.flank_kl_z);
      const yOfZ = (z) => plotBot - Math.min(1.0, Math.max(0.0, z / maxZ)) * (plotH - 6);

      // Horizontal Y ticks & labels
      ctx.fillStyle = "#222222";
      ctx.font = "10px Consolas, monospace";
      ctx.textAlign = "right";
      [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
        const y = plotBot - f * (plotH - 6);
        const val = (f * maxZ).toFixed(1);
        ctx.fillText(val, padL - 4, y + 3);
        ctx.strokeStyle = "#e6e6e6";
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
      });

      // Shaded gray permutation-null band [0 .. threshold_z * 0.5] (like RDP5's gray CI band!)
      const yBandTop = yOfZ(Math.min(maxZ, prof.threshold_z));
      ctx.fillStyle = "rgba(180, 180, 180, 0.25)";
      ctx.fillRect(padL, yBandTop, plotW, plotBot - yBandTop);

      // Dashed threshold line
      ctx.strokeStyle = "#444444";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, yBandTop);
      ctx.lineTo(padL + plotW, yBandTop);
      ctx.stroke();
      ctx.setLineDash([]);

      // Filled gray area + black line for flank_kl z-score (matching RDP5's bottom-left plot!)
      ctx.fillStyle = "rgba(110, 110, 110, 0.30)";
      ctx.beginPath();
      ctx.moveTo(xOfBp(prof.positions_orig_bp[0]), plotBot);
      prof.positions_orig_bp.forEach((pos, idx) => {
        ctx.lineTo(xOfBp(pos), yOfZ(prof.flank_kl_z[idx]));
      });
      ctx.lineTo(xOfBp(prof.positions_orig_bp[prof.positions_orig_bp.length - 1]), plotBot);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      prof.positions_orig_bp.forEach((pos, idx) => {
        const x = xOfBp(pos);
        const y = yOfZ(prof.flank_kl_z[idx]);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Secondary Transformer P(bp) trace
      ctx.strokeStyle = "#762a83";
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      prof.positions_orig_bp.forEach((pos, idx) => {
        const x = xOfBp(pos);
        const y = plotBot - Math.min(1.0, Math.max(0.0, prof.tf_prob[idx])) * (plotH - 6);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // X-axis ticks (Position in alignment)
    ctx.fillStyle = "#111111";
    ctx.strokeStyle = "#222222";
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "center";
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
      const bp = Math.round((L * i) / nTicks);
      const tx = xOfBp(bp);
      ctx.beginPath();
      ctx.moveTo(tx, plotBot);
      ctx.lineTo(tx, plotBot + 4);
      ctx.stroke();
      ctx.fillText(fmtInt(bp), tx, plotBot + 15);
    }
    ctx.fillText("Position in alignment (bp)", padL + plotW / 2, height - 3);
  }

  function handleCanvasHover(e) {
    if (!state.currentRun) return;
    const canvas = el("genomeTrackCanvas");
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Update RDP5 cursor readout box (bp : z)
    const padL = 46;
    const padR = 14;
    const plotW = Math.max(1, rect.width - padL - padR);
    const L = state.currentRun.alignment.genome_length_bp;
    const bpAtCursor = Math.max(0, Math.min(L, Math.round(((mx - padL) / plotW) * L)));
    const prof = state.currentRun.track_profile;
    let zAtCursor = 0;
    if (prof && prof.positions_orig_bp && prof.positions_orig_bp.length > 0) {
      let bestI = 0;
      let bestD = Infinity;
      prof.positions_orig_bp.forEach((p, idx) => {
        const d = Math.abs(p - bpAtCursor);
        if (d < bestD) {
          bestD = d;
          bestI = idx;
        }
      });
      zAtCursor = prof.flank_kl_z[bestI] || 0;
    }
    el("cursorReadoutBox").textContent = `${bpAtCursor} : ${zAtCursor.toFixed(2)}`;

    const hit = state.canvasMarkers.find((m) => Math.abs(m.x - mx) <= 9 && Math.abs(m.y - my) <= 12);
    const tip = el("trackTooltip");
    if (hit) {
      tip.innerHTML = `<strong>${escapeHtml(hit.label)}</strong><br/>${escapeHtml(hit.detail)}`;
      tip.style.left = `${Math.min(rect.width - 240, Math.max(6, mx + 10))}px`;
      tip.style.top = `${Math.max(6, my + 12)}px`;
      tip.classList.remove("hidden");
    } else {
      tip.classList.add("hidden");
    }
  }

  function handleCanvasClick(e) {
    if (!state.currentRun) return;
    const canvas = el("genomeTrackCanvas");
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const hit = state.canvasMarkers.find(
      (m) => m.kind === "pred" && Math.abs(m.x - mx) <= 12 && Math.abs(m.y - my) <= 15
    );
    if (hit && hit.eventId) {
      selectEvent(hit.eventId, true);
    }
  }

  // -------------------------------------------------------------------------
  // Canvas 3: Bottom-Right RDP5 Mosaic Sequence Map
  // -------------------------------------------------------------------------
  function colorForParent(idx) {
    return PALETTE[Math.abs(Number(idx)) % PALETTE.length];
  }

  function drawMosaicMap(run) {
    const canvas = el("mosaicMapCanvas");
    const dpr = window.devicePixelRatio || 1;
    const wrap = canvas.parentElement;
    const width = Math.max(460, Math.floor(wrap.clientWidth - 10));
    const L = Math.max(1, run.alignment.genome_length_bp);

    const selectedOnly = el("chkMosaicSelectedOnly").checked;
    let evs = run.consolidated_events || [];
    if (selectedOnly && state.selectedEventId !== null) {
      evs = evs.filter((x) => x.event_id === state.selectedEventId);
    }

    // Collect rows: one bar per (event, carrier) up to 14 rows for clean display
    const bars = [];
    evs.forEach((ev) => {
      (ev.recombinants || []).slice(0, 4).forEach((r) => {
        bars.push({ ev, r });
      });
    });

    el("mosaicLegendSummary").textContent = `${bars.length} carrier tract(s) shown`;
    const rowH = 26;
    const height = Math.max(200, 26 + bars.length * rowH + 16);

    canvas.style.height = `${height}px`;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const padL = 12;
    const padR = 14;
    const plotW = width - padL - padR;
    const xOfBp = (bp) => padL + (Math.max(0, Math.min(L, bp)) / L) * plotW;

    if (bars.length === 0) {
      ctx.fillStyle = "#555555";
      ctx.font = "12px Consolas, monospace";
      ctx.fillText("No recombinant carrier tracts detected in this run.", 14, 30);
      return;
    }

    bars.forEach(({ ev, r }, idx) => {
      const y = 14 + idx * rowH;
      const bp = ev.most_plausible_bp_orig;
      const xBp = xOfBp(bp);
      const xSpanL = xOfBp(ev.span_orig_start);
      const xSpanR = xOfBp(ev.span_orig_end);
      const colL = colorForParent(r.ext_parent_left);
      const colR = colorForParent(r.ext_parent_right);
      const isSel = ev.event_id === state.selectedEventId;

      // Label above bar (like RDP5's sequence name line right above each colored tract bar)
      ctx.fillStyle = isSel ? "#000000" : "#333333";
      ctx.font = isSel ? "bold 10.5px Consolas, monospace" : "10.5px Consolas, monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        `Ev#${ev.event_id} Seq ${r.seq_idx} (${r.seq_name.slice(0, 18)}) | P_L=${r.ext_parent_left} (${(r.left_sim * 100).toFixed(1)}%) → P_R=${r.ext_parent_right} (${(r.right_sim * 100).toFixed(1)}%) @ ${bp} bp`,
        padL,
        y + 8
      );

      // Left parental tract bar [0 .. bp]
      ctx.fillStyle = colL;
      ctx.fillRect(padL, y + 11, Math.max(2, xBp - padL), 10);

      // Right parental tract bar [bp .. L]
      ctx.fillStyle = colR;
      ctx.fillRect(xBp, y + 11, Math.max(2, padL + plotW - xBp), 10);

      // Uncertainty span box
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 1;
      ctx.strokeRect(xSpanL, y + 10, Math.max(3, xSpanR - xSpanL), 12);

      // Breakpoint divider tick
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xBp, y + 9);
      ctx.lineTo(xBp, y + 23);
      ctx.stroke();
    });
  }

  // -------------------------------------------------------------------------
  // Annotations & Exports
  // -------------------------------------------------------------------------
  async function loadAnnotations() {
    const res = await fetch("/api/annotations");
    const data = await res.json();
    state.annotations = data.annotations || [];
    el("annotationCountBadge").textContent = String(state.annotations.length);
    const tbody = el("annotationsBody");
    if (state.annotations.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6">No notes saved yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = state.annotations
      .map(
        (a) =>
          `<tr>` +
          `<td class="mono">${escapeHtml((a.created_at_iso || "").slice(0, 19).replace("T", " "))}</td>` +
          `<td class="mono">${escapeHtml(a.dataset_name)}</td>` +
          `<td class="mono">${escapeHtml(a.target_id)}</td>` +
          `<td class="mono">${escapeHtml(a.verdict)}</td>` +
          `<td>${escapeHtml(a.reviewer_name)}</td>` +
          `<td>${escapeHtml(a.comment)}</td>` +
          `</tr>`
      )
      .join("");
  }

  function openAnnotationModal(targetType, targetId) {
    el("annTargetType").value = targetType || "predicted_event";
    el("annTargetId").value = targetId || "";
    el("annComment").value = "";
    el("annotationModal").classList.remove("hidden");
  }

  function closeAnnotationModal() {
    el("annotationModal").classList.add("hidden");
  }

  async function handleSaveAnnotation(e) {
    e.preventDefault();
    const run = state.currentRun;
    const payload = {
      dataset_name: run ? run.alignment.dataset_name : "unknown",
      run_id: run ? run.timings.run_id : "manual",
      target_type: el("annTargetType").value,
      target_id: el("annTargetId").value.trim(),
      verdict: el("annVerdict").value,
      reviewer_name: "Darren Martin",
      comment: el("annComment").value.trim(),
    };
    const res = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      closeAnnotationModal();
      await loadAnnotations();
      switchTab("tab-notes");
    }
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function exportOpenRdpTsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    const mode = state.currentRun.config.scoring_mode;
    const lines = ["Method\tStart\tEnd\tBreakpoint\tRecombinant\tParent1\tParent2\tFlankKL_Z\tTF_Logit"];
    (state.currentRun.consolidated_events || []).forEach((ev) => {
      const r = ev.recombinants && ev.recombinants[0];
      lines.push(
        [
          mode,
          ev.span_orig_start,
          ev.span_orig_end,
          ev.most_plausible_bp_orig,
          r ? r.seq_name : "-",
          r ? r.ext_parent_left_name : "-",
          r ? r.ext_parent_right_name : "-",
          ev.score_z.toFixed(4),
          ev.tf_logit.toFixed(4),
        ].join("\t")
      );
    });
    downloadBlob(`${ds}_openrdp_summary.tsv`, lines.join("\n"), "text/tab-separated-values");
  }

  function exportBreakpointsCsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    const header = [
      "event_id",
      "most_plausible_bp_orig",
      "span_orig_start",
      "span_orig_end",
      "n_peaks_consolidated",
      "score_z",
      "tf_logit",
      "tf_prob",
      "boundary_tx",
      "n_switch_recombinants",
      "recombinant_sequences",
      "match_status",
      "matched_gt_bp",
      "bp_error",
    ];
    const rows = (state.currentRun.consolidated_events || []).map((ev) => [
      ev.event_id,
      ev.most_plausible_bp_orig,
      ev.span_orig_start,
      ev.span_orig_end,
      ev.n_peaks_consolidated,
      ev.score_z.toFixed(4),
      ev.tf_logit.toFixed(4),
      ev.tf_prob.toFixed(4),
      ev.boundary_tx,
      ev.n_switch_recombinants,
      ev.recombinant_sequences.join(";"),
      ev.match_status,
      ev.matched_gt_bp ?? "",
      ev.bp_error ?? "",
    ]);
    downloadBlob(`${ds}_breakpoints_consolidated.csv`, [header.join(","), ...rows.map((r) => r.join(","))].join("\n"), "text/csv");
  }

  function exportSequenceCallsCsv() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    const header = [
      "event_id",
      "most_plausible_bp_orig",
      "span_orig_start",
      "span_orig_end",
      "seq_idx",
      "seq_name",
      "p_switch",
      "is_recombinant",
      "parent_left",
      "parent_right",
      "ext_parent_left",
      "ext_parent_right",
      "left_sim",
      "right_sim",
    ];
    const rows = [];
    (state.currentRun.consolidated_events || []).forEach((ev) => {
      (ev.recombinants || []).forEach((r) => {
        rows.push([
          ev.event_id,
          ev.most_plausible_bp_orig,
          ev.span_orig_start,
          ev.span_orig_end,
          r.seq_idx,
          `"${String(r.seq_name).replace(/"/g, '""')}"`,
          r.p_switch.toFixed(4),
          r.is_recombinant,
          r.parent_left,
          r.parent_right,
          r.ext_parent_left,
          r.ext_parent_right,
          r.left_sim.toFixed(4),
          r.right_sim.toFixed(4),
        ]);
      });
    });
    downloadBlob(`${ds}_sequence_calls_consolidated.csv`, [header.join(","), ...rows.map((r) => r.join(","))].join("\n"), "text/csv");
  }

  function exportFullJson() {
    if (!state.currentRun) return;
    const ds = state.currentRun.alignment.dataset_name;
    downloadBlob(`${ds}_diagnostic_report.json`, JSON.stringify(state.currentRun, null, 2), "application/json");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
