import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { SymptomFormData } from "@/components/SymptomForm";

interface PredictionData {
  condition: string;
  confidence: number;
  risk_level: "low" | "moderate" | "high";
  description: string;
  clinical_reasoning: string;
  recommendations: string[];
  when_to_see_doctor: string;
  differential_diagnoses: { name: string; likelihood: string }[];
  preprocessing_steps: string[];
  model_comparison: {
    xgboost: { accuracy: number; model: string };
    random_forest: { accuracy: number; model: string };
    logistic_regression: { accuracy: number; model: string };
    decision_tree: { accuracy: number; model: string };
  };
  feature_importance: { feature: string; importance: number }[];
  probability_distribution: { condition: string; probability: number }[];
  primary_model: string;
  pipeline_steps: string[];
  report_findings?: string;
  reports_analyzed?: number;
}

const COLORS = {
  primary: [13, 121, 125] as [number, number, number],
  primaryLight: [230, 247, 247] as [number, number, number],
  primaryMid: [180, 225, 225] as [number, number, number],
  text: [30, 41, 59] as [number, number, number],
  muted: [100, 116, 139] as [number, number, number],
  white: [255, 255, 255] as [number, number, number],
  low: [34, 139, 34] as [number, number, number],
  lowBg: [240, 253, 244] as [number, number, number],
  moderate: [218, 165, 32] as [number, number, number],
  moderateBg: [254, 252, 232] as [number, number, number],
  high: [220, 53, 69] as [number, number, number],
  highBg: [254, 242, 242] as [number, number, number],
  border: [226, 232, 240] as [number, number, number],
  bgLight: [248, 250, 252] as [number, number, number],
  chartColors: [
    [13, 121, 125],
    [45, 160, 120],
    [52, 144, 205],
    [218, 165, 32],
    [220, 100, 69],
    [140, 80, 180],
    [70, 180, 160],
    [200, 120, 50],
  ] as [number, number, number][],
};

const SYMPTOM_LABELS: Record<string, string> = {
  nasal_congestion: "Nasal Congestion / Blockage",
  facial_pain: "Facial Pain / Pressure",
  nasal_discharge: "Nasal Discharge",
  reduced_smell: "Reduced Sense of Smell",
  headache: "Headache",
  post_nasal_drip: "Post-Nasal Drip",
  cough: "Cough",
  ear_pressure: "Ear Pressure / Fullness",
  itchy_watery_eyes: "Itchy / Watery Eyes",
  fever: "Fever",
  fatigue: "Fatigue / Malaise",
  dental_pain: "Upper Dental Pain",
  bad_breath: "Bad Breath (Halitosis)",
  snoring: "Snoring / Sleep Issues",
  nosebleeds: "Nosebleeds",
};

const DURATION_LABELS: Record<string, string> = {
  less_than_week: "Less than 1 week",
  "1_to_4_weeks": "1–4 weeks",
  "1_to_3_months": "1–3 months",
  more_than_3_months: "More than 3 months",
};

const ENVIRONMENT_LABELS: Record<string, string> = {
  urban_pollution: "Urban / High Pollution",
  rural_clean: "Rural / Clean Air",
  industrial: "Industrial Area",
  dry_climate: "Dry / Arid Climate",
  humid_climate: "Humid / Tropical",
};

function getRiskColor(level: string): [number, number, number] {
  if (level === "low") return COLORS.low;
  if (level === "moderate") return COLORS.moderate;
  return COLORS.high;
}

function getRiskBg(level: string): [number, number, number] {
  if (level === "low") return COLORS.lowBg;
  if (level === "moderate") return COLORS.moderateBg;
  return COLORS.highBg;
}

function addSectionHeader(doc: jsPDF, y: number, title: string): number {
  if (y > 258) {
    doc.addPage();
    y = 20;
  }
  doc.setFillColor(...COLORS.primary);
  doc.roundedRect(14, y, 3, 10, 1, 1, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(...COLORS.text);
  doc.text(title, 22, y + 7.5);
  return y + 16;
}

function checkPageBreak(doc: jsPDF, y: number, needed: number): number {
  if (y + needed > 275) {
    doc.addPage();
    return 20;
  }
  return y;
}

/** Draw a horizontal bar chart */
function drawBarChart(
  doc: jsPDF,
  y: number,
  items: { label: string; value: number; maxValue?: number }[],
  pageWidth: number,
  options: { suffix?: string; barHeight?: number; labelWidth?: number } = {}
): number {
  const { suffix = "%", barHeight = 7, labelWidth = 55 } = options;
  const chartLeft = 22 + labelWidth;
  const chartRight = pageWidth - 22;
  const chartWidth = chartRight - chartLeft;
  const maxVal = items.reduce((m, item) => Math.max(m, item.maxValue ?? item.value), 0) || 100;

  items.forEach((item, i) => {
    const rowY = y + i * (barHeight + 4);
    // Label
    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...COLORS.text);
    const truncLabel = item.label.length > 22 ? item.label.substring(0, 20) + "…" : item.label;
    doc.text(truncLabel, 22, rowY + barHeight / 2 + 1);

    // Background bar
    doc.setFillColor(...COLORS.border);
    doc.roundedRect(chartLeft, rowY, chartWidth, barHeight, 2, 2, "F");

    // Value bar
    const barW = Math.max(2, (item.value / maxVal) * chartWidth);
    const color = COLORS.chartColors[i % COLORS.chartColors.length];
    doc.setFillColor(...color);
    doc.roundedRect(chartLeft, rowY, barW, barHeight, 2, 2, "F");

    // Value label
    doc.setFontSize(7);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...COLORS.white);
    if (barW > 20) {
      doc.text(`${item.value}${suffix}`, chartLeft + barW - 3, rowY + barHeight / 2 + 1, { align: "right" });
    } else {
      doc.setTextColor(...COLORS.text);
      doc.text(`${item.value}${suffix}`, chartLeft + barW + 3, rowY + barHeight / 2 + 1);
    }
  });

  return y + items.length * (barHeight + 4) + 4;
}

export function generateMedicalReport(result: PredictionData, formData?: SymptomFormData) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  const reportId = `SR-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;

  // ── Header ──
  doc.setFillColor(...COLORS.primary);
  doc.rect(0, 0, pageWidth, 40, "F");
  // Subtle accent stripe
  doc.setFillColor(255, 255, 255);
  doc.setGState(new (doc as any).GState({ opacity: 0.08 }));
  doc.rect(0, 36, pageWidth, 4, "F");
  doc.setGState(new (doc as any).GState({ opacity: 1 }));

  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(...COLORS.white);
  doc.text("SinusAI Medical Report", 16, 16);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(200, 230, 230);
  doc.text("AI-Based Sinus & Nasal Disease Prediction System", 16, 24);
  doc.text(`Report ID: ${reportId}`, 16, 31);
  doc.text(`Generated: ${dateStr}`, 16, 37);

  doc.setFontSize(8);
  doc.setTextColor(180, 220, 220);
  doc.text("ML-Powered  •  XGBoost + AI Clinical Validation", pageWidth - 16, 37, { align: "right" });

  let y = 50;

  // ── Primary Diagnosis Box ──
  const riskColor = getRiskColor(result.risk_level);
  const riskBg = getRiskBg(result.risk_level);
  doc.setFillColor(...riskBg);
  doc.setDrawColor(...riskColor);
  doc.setLineWidth(0.6);
  doc.roundedRect(14, y, pageWidth - 28, 38, 3, 3, "FD");

  // Risk badge
  doc.setFillColor(...riskColor);
  doc.roundedRect(20, y + 5, 32, 7, 2, 2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.white);
  doc.text(result.risk_level.toUpperCase() + " RISK", 36, y + 10, { align: "center" });

  // Confidence badge
  doc.setFillColor(...COLORS.primary);
  doc.roundedRect(56, y + 5, 28, 7, 2, 2, "F");
  doc.text(`${result.confidence}% CONF.`, 70, y + 10, { align: "center" });

  if (result.primary_model) {
    doc.setFontSize(7);
    doc.setTextColor(...COLORS.muted);
    doc.text(`Model: ${result.primary_model}`, pageWidth - 20, y + 10, { align: "right" });
  }

  doc.setFontSize(16);
  doc.setTextColor(...COLORS.text);
  doc.text(result.condition, 20, y + 22);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.muted);
  const descLines = doc.splitTextToSize(result.description, pageWidth - 44);
  doc.text(descLines.slice(0, 2), 20, y + 29);

  y += 46;

  // ── Confidence Progress Bar ──
  y = checkPageBreak(doc, y, 14);
  doc.setFillColor(...COLORS.border);
  doc.roundedRect(14, y, pageWidth - 28, 5, 2, 2, "F");
  doc.setFillColor(...riskColor);
  const barWidth = ((pageWidth - 28) * result.confidence) / 100;
  doc.roundedRect(14, y, barWidth, 5, 2, 2, "F");
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.muted);
  doc.text("Prediction Confidence", 14, y + 10);
  doc.text(`${result.confidence}%`, pageWidth - 14, y + 10, { align: "right" });
  y += 16;

  // ══════════════════════════════════════════════
  // ── NEW: Patient Demographics & Symptom Summary ──
  // ══════════════════════════════════════════════
  if (formData) {
    y = checkPageBreak(doc, y, 50);
    y = addSectionHeader(doc, y, "Patient Profile & Symptom Breakdown");

    // Demographics row
    doc.setFillColor(...COLORS.bgLight);
    doc.roundedRect(22, y, pageWidth - 44, 24, 2, 2, "F");

    const demoItems = [
      { label: "Age", value: formData.age },
      { label: "Gender", value: formData.gender.charAt(0).toUpperCase() + formData.gender.slice(1) },
      { label: "Duration", value: DURATION_LABELS[formData.duration] || formData.duration },
      { label: "Severity", value: formData.severity.charAt(0).toUpperCase() + formData.severity.slice(1) },
    ];

    const colWidth = (pageWidth - 44) / demoItems.length;
    demoItems.forEach((item, i) => {
      const cx = 22 + colWidth * i + colWidth / 2;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(7);
      doc.setTextColor(...COLORS.muted);
      doc.text(item.label, cx, y + 8, { align: "center" });
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...COLORS.text);
      doc.text(item.value || "—", cx, y + 16, { align: "center" });
    });
    y += 30;

    // Risk factors row
    const riskFactors: string[] = [];
    if (formData.allergies) riskFactors.push("Known Allergies");
    if (formData.smoking) riskFactors.push("Smoker / Ex-smoker");
    if (formData.previous_sinus_history) riskFactors.push("Previous Sinus History");
    if (formData.environment) riskFactors.push(ENVIRONMENT_LABELS[formData.environment] || formData.environment);
    if (formData.medications) riskFactors.push(`Medications: ${formData.medications}`);

    if (riskFactors.length > 0) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...COLORS.text);
      doc.text("Risk Factors:", 22, y);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(...COLORS.muted);
      const rfText = riskFactors.join("  •  ");
      const rfLines = doc.splitTextToSize(rfText, pageWidth - 44);
      doc.text(rfLines, 22, y + 5);
      y += 5 + rfLines.length * 4 + 4;
    }

    // Symptom checklist as visual grid
    y = checkPageBreak(doc, y, 10 + Math.ceil(Object.keys(SYMPTOM_LABELS).length / 2) * 6);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.text);
    doc.text("Reported Symptoms:", 22, y);
    y += 5;

    const allSymptomIds = Object.keys(SYMPTOM_LABELS);
    const colW = (pageWidth - 44) / 2;

    allSymptomIds.forEach((id, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const sx = 22 + col * colW;
      const sy = y + row * 6;

      if (col === 0) {
        // Check page break at start of each row
        const needed = 6;
        if (sy + needed > 275) {
          doc.addPage();
          y = 20 - row * 6; // adjust base y
        }
      }

      const actualSy = y + row * 6;
      const isSelected = formData.symptoms.includes(id);

      // Checkbox
      if (isSelected) {
        doc.setFillColor(...COLORS.primary);
        doc.roundedRect(sx, actualSy - 2.5, 3.5, 3.5, 0.5, 0.5, "F");
        doc.setFont("helvetica", "bold");
        doc.setFontSize(6);
        doc.setTextColor(...COLORS.white);
        doc.text("✓", sx + 0.7, actualSy + 0.2);
      } else {
        doc.setDrawColor(...COLORS.border);
        doc.setLineWidth(0.3);
        doc.roundedRect(sx, actualSy - 2.5, 3.5, 3.5, 0.5, 0.5, "S");
      }

      doc.setFont("helvetica", isSelected ? "bold" : "normal");
      doc.setFontSize(8);
      doc.setTextColor(...(isSelected ? COLORS.text : COLORS.muted));
      doc.text(SYMPTOM_LABELS[id], sx + 5, actualSy);
    });

    y += Math.ceil(allSymptomIds.length / 2) * 6 + 6;
  }

  // ── ML Pipeline Steps ──
  if (result.pipeline_steps?.length) {
    y = checkPageBreak(doc, y, 16);
    y = addSectionHeader(doc, y, "ML Pipeline Executed");
    doc.setFontSize(8);
    doc.setTextColor(...COLORS.muted);
    const stepsText = result.pipeline_steps.map((s, i) => `${i + 1}. ${s}`).join("   →   ");
    const stepsLines = doc.splitTextToSize(stepsText, pageWidth - 36);
    doc.text(stepsLines, 22, y);
    y += stepsLines.length * 4.5 + 6;
  }

  // ══════════════════════════════════════════════
  // ── NEW: Model Comparison Chart ──
  // ══════════════════════════════════════════════
  if (result.model_comparison) {
    const models = [
      result.model_comparison.xgboost,
      result.model_comparison.random_forest,
      result.model_comparison.logistic_regression,
      result.model_comparison.decision_tree,
    ];
    y = checkPageBreak(doc, y, 60);
    y = addSectionHeader(doc, y, "Model Performance Comparison");

    y = drawBarChart(
      doc,
      y,
      models.map((m) => ({ label: m.model, value: m.accuracy, maxValue: 100 })),
      pageWidth,
      { suffix: "%", barHeight: 8, labelWidth: 50 }
    );

    // Mini legend
    doc.setFontSize(7);
    doc.setTextColor(...COLORS.muted);
    doc.text(`★ Primary: ${models[0].model} (${models[0].accuracy}%)`, 22, y + 2);
    y += 10;
  }

  // ══════════════════════════════════════════════
  // ── NEW: Feature Importance Chart ──
  // ══════════════════════════════════════════════
  if (result.feature_importance?.length) {
    const fiCount = result.feature_importance.length;
    y = checkPageBreak(doc, y, 20 + fiCount * 12);
    y = addSectionHeader(doc, y, "Feature Importance (Explainability)");

    y = drawBarChart(
      doc,
      y,
      result.feature_importance.map((f) => ({
        label: f.feature,
        value: Math.round(f.importance * 100) / 100,
        maxValue: 1,
      })),
      pageWidth,
      { suffix: "", barHeight: 7, labelWidth: 50 }
    );
    y += 4;
  }

  // ══════════════════════════════════════════════
  // ── NEW: Disease Probability Chart ──
  // ══════════════════════════════════════════════
  if (result.probability_distribution?.length) {
    y = checkPageBreak(doc, y, 20 + result.probability_distribution.length * 12);
    y = addSectionHeader(doc, y, "Disease Probability Distribution");

    y = drawBarChart(
      doc,
      y,
      result.probability_distribution.map((d) => ({
        label: d.condition,
        value: d.probability,
        maxValue: 100,
      })),
      pageWidth,
      { suffix: "%", barHeight: 7, labelWidth: 55 }
    );
    y += 4;
  }

  // ══════════════════════════════════════════════
  // ── NEW: Doctor-Ready Clinical Summary ──
  // ══════════════════════════════════════════════
  y = checkPageBreak(doc, y, 60);
  doc.addPage();
  y = 20;
  y = addSectionHeader(doc, y, "Clinical Summary for Healthcare Provider");

  doc.setFillColor(...COLORS.primaryLight);
  doc.setDrawColor(...COLORS.primary);
  doc.setLineWidth(0.4);
  const summaryStartY = y;

  // Build summary text
  const summaryParts: string[] = [];
  summaryParts.push(`DIAGNOSIS: ${result.condition} (${result.confidence}% confidence, ${result.risk_level} risk)`);
  summaryParts.push("");

  if (formData) {
    summaryParts.push(`PATIENT: ${formData.age}-year-old ${formData.gender}, symptoms for ${DURATION_LABELS[formData.duration] || formData.duration}, severity: ${formData.severity}`);
    const selectedSymptomNames = formData.symptoms.map((id) => SYMPTOM_LABELS[id] || id).join(", ");
    summaryParts.push(`PRESENTING SYMPTOMS: ${selectedSymptomNames}`);

    const factors: string[] = [];
    if (formData.allergies) factors.push("allergy history");
    if (formData.smoking) factors.push("smoking history");
    if (formData.previous_sinus_history) factors.push("previous sinus conditions");
    if (formData.environment) factors.push(`environmental: ${ENVIRONMENT_LABELS[formData.environment] || formData.environment}`);
    if (factors.length > 0) summaryParts.push(`RISK FACTORS: ${factors.join(", ")}`);
    if (formData.medications) summaryParts.push(`CURRENT MEDICATIONS: ${formData.medications}`);
    summaryParts.push("");
  }

  if (result.clinical_reasoning) {
    summaryParts.push(`AI CLINICAL REASONING: ${result.clinical_reasoning}`);
    summaryParts.push("");
  }

  if (result.differential_diagnoses?.length) {
    summaryParts.push(`DIFFERENTIAL DIAGNOSES: ${result.differential_diagnoses.map((d) => `${d.name} (${d.likelihood})`).join("; ")}`);
    summaryParts.push("");
  }

  if (result.recommendations?.length) {
    summaryParts.push("RECOMMENDATIONS:");
    result.recommendations.forEach((r, i) => summaryParts.push(`  ${i + 1}. ${r}`));
    summaryParts.push("");
  }

  if (result.when_to_see_doctor) {
    summaryParts.push(`REFERRAL GUIDANCE: ${result.when_to_see_doctor}`);
  }

  const fullSummary = summaryParts.join("\n");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.text);
  const summaryLines = doc.splitTextToSize(fullSummary, pageWidth - 48);

  // Draw background box
  const summaryHeight = summaryLines.length * 4 + 8;
  const adjustedSummaryHeight = Math.min(summaryHeight, 240);
  doc.roundedRect(22, summaryStartY, pageWidth - 44, adjustedSummaryHeight, 2, 2, "FD");

  // Print summary, handling page breaks
  let textY = summaryStartY + 5;
  summaryLines.forEach((line: string) => {
    if (textY > 272) {
      doc.addPage();
      textY = 20;
    }
    // Bold for section headers
    if (line.match(/^[A-Z ]+:/) && !line.startsWith("  ")) {
      doc.setFont("helvetica", "bold");
    } else {
      doc.setFont("helvetica", "normal");
    }
    doc.text(line, 26, textY);
    textY += 4;
  });
  y = textY + 6;

  // ── Report Findings (if uploaded) ──
  if (result.report_findings && result.report_findings.trim()) {
    y = checkPageBreak(doc, y, 30);
    y = addSectionHeader(doc, y, `Medical Report Findings (${result.reports_analyzed || 0} analyzed)`);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.text);
    const findLines = doc.splitTextToSize(result.report_findings, pageWidth - 44);
    findLines.forEach((line: string) => {
      y = checkPageBreak(doc, y, 5);
      doc.text(line, 22, y);
      y += 4;
    });
    y += 6;
  }

  // ── Data Preprocessing Applied ──
  if (result.preprocessing_steps?.length) {
    y = checkPageBreak(doc, y, 20);
    y = addSectionHeader(doc, y, "Data Preprocessing Applied");
    doc.setFontSize(8);
    doc.setTextColor(...COLORS.muted);
    result.preprocessing_steps.forEach((step, i) => {
      y = checkPageBreak(doc, y, 6);
      doc.text(`${i + 1}. ${step}`, 22, y);
      y += 5;
    });
    y += 4;
  }

  // ── Footer Disclaimer ──
  y = checkPageBreak(doc, y, 30);
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.3);
  doc.line(14, y, pageWidth - 14, y);
  y += 6;
  doc.setFillColor(...COLORS.bgLight);
  doc.roundedRect(14, y, pageWidth - 28, 22, 2, 2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.muted);
  doc.text("DISCLAIMER", 18, y + 5);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(6.5);
  const disclaimer =
    "This report is generated by an ML model (XGBoost) validated by AI clinical reasoning. It is for informational and educational purposes only and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for proper medical evaluation. This report is intended to support—not replace—clinical decision making.";
  const disclaimerLines = doc.splitTextToSize(disclaimer, pageWidth - 36);
  doc.text(disclaimerLines, 18, y + 10);

  // ── Page numbers & footers ──
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(...COLORS.muted);
    doc.text(`Page ${i} of ${totalPages}`, pageWidth / 2, 290, { align: "center" });
    doc.text("SinusAI Medical Report — Confidential", 14, 290);
    doc.text(reportId, pageWidth - 14, 290, { align: "right" });
  }

  doc.save(`SinusAI-Report-${reportId}.pdf`);
}
