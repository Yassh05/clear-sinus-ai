import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

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
}

const COLORS = {
  primary: [13, 121, 125] as [number, number, number],
  primaryLight: [230, 247, 247] as [number, number, number],
  text: [30, 41, 59] as [number, number, number],
  muted: [100, 116, 139] as [number, number, number],
  white: [255, 255, 255] as [number, number, number],
  low: [34, 139, 34] as [number, number, number],
  moderate: [218, 165, 32] as [number, number, number],
  high: [220, 53, 69] as [number, number, number],
  border: [226, 232, 240] as [number, number, number],
  bgLight: [248, 250, 252] as [number, number, number],
};

function getRiskColor(level: string): [number, number, number] {
  if (level === "low") return COLORS.low;
  if (level === "moderate") return COLORS.moderate;
  return COLORS.high;
}

function addSectionHeader(doc: jsPDF, y: number, title: string): number {
  if (y > 260) {
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

export function generateMedicalReport(result: PredictionData) {
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
  doc.rect(0, 0, pageWidth, 38, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.setTextColor(...COLORS.white);
  doc.text("SinusAI Medical Report", 16, 16);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(200, 230, 230);
  doc.text("AI-Based Sinus & Nasal Disease Prediction System", 16, 23);
  doc.text(`Report ID: ${reportId}  |  Generated: ${dateStr}`, 16, 30);

  doc.setFontSize(8);
  doc.text("ML-Powered  •  XGBoost + AI Clinical Validation", pageWidth - 16, 30, { align: "right" });

  let y = 48;

  // ── Primary Diagnosis ──
  const riskColor = getRiskColor(result.risk_level);
  doc.setFillColor(...(result.risk_level === "low" ? [240, 253, 244] : result.risk_level === "moderate" ? [254, 252, 232] : [254, 242, 242]) as [number, number, number]);
  doc.setDrawColor(...riskColor);
  doc.setLineWidth(0.5);
  doc.roundedRect(14, y, pageWidth - 28, 36, 3, 3, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(...riskColor);
  doc.text(`${result.risk_level.toUpperCase()} RISK  •  ${result.confidence}% CONFIDENCE`, 20, y + 8);

  doc.setFontSize(16);
  doc.setTextColor(...COLORS.text);
  doc.text(result.condition, 20, y + 18);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.muted);
  const descLines = doc.splitTextToSize(result.description, pageWidth - 44);
  doc.text(descLines.slice(0, 2), 20, y + 26);

  if (result.primary_model) {
    doc.setFontSize(7);
    doc.text(`Primary Model: ${result.primary_model}`, pageWidth - 20, y + 8, { align: "right" });
  }

  y += 44;

  // ── Confidence Bar ──
  y = checkPageBreak(doc, y, 16);
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

  // ── ML Pipeline Steps ──
  if (result.pipeline_steps?.length) {
    y = addSectionHeader(doc, y, "ML Pipeline Executed");
    doc.setFontSize(8);
    doc.setTextColor(...COLORS.muted);
    const stepsText = result.pipeline_steps.map((s, i) => `${i + 1}. ${s}`).join("   →   ");
    const stepsLines = doc.splitTextToSize(stepsText, pageWidth - 36);
    doc.text(stepsLines, 22, y);
    y += stepsLines.length * 4.5 + 6;
  }

  // ── Model Comparison ──
  if (result.model_comparison) {
    y = checkPageBreak(doc, y, 40);
    y = addSectionHeader(doc, y, "Model Comparison");
    const models = [
      result.model_comparison.xgboost,
      result.model_comparison.random_forest,
      result.model_comparison.logistic_regression,
      result.model_comparison.decision_tree,
    ];
    autoTable(doc, {
      startY: y,
      margin: { left: 22, right: 22 },
      head: [["Model", "Accuracy (%)", "Status"]],
      body: models.map((m, i) => [m.model, `${m.accuracy}%`, i === 0 ? "★ Primary" : ""]),
      headStyles: { fillColor: COLORS.primary, fontSize: 8, font: "helvetica" },
      bodyStyles: { fontSize: 8, textColor: COLORS.text },
      alternateRowStyles: { fillColor: COLORS.bgLight },
      columnStyles: { 2: { textColor: COLORS.primary, fontStyle: "bold" } },
    });
    y = (doc as any).lastAutoTable.finalY + 8;
  }

  // ── Feature Importance ──
  if (result.feature_importance?.length) {
    y = checkPageBreak(doc, y, 40);
    y = addSectionHeader(doc, y, "Feature Importance (Explainability)");
    autoTable(doc, {
      startY: y,
      margin: { left: 22, right: 22 },
      head: [["Feature", "Importance Score", "Visual"]],
      body: result.feature_importance.map((f) => {
        const barLen = Math.round(f.importance * 20);
        const bar = "█".repeat(barLen) + "░".repeat(20 - barLen);
        return [f.feature, f.importance.toFixed(3), bar];
      }),
      headStyles: { fillColor: COLORS.primary, fontSize: 8 },
      bodyStyles: { fontSize: 8, textColor: COLORS.text },
      alternateRowStyles: { fillColor: COLORS.bgLight },
      columnStyles: { 2: { fontStyle: "bold", textColor: COLORS.primary } },
    });
    y = (doc as any).lastAutoTable.finalY + 8;
  }

  // ── Disease Probability Distribution ──
  if (result.probability_distribution?.length) {
    y = checkPageBreak(doc, y, 40);
    y = addSectionHeader(doc, y, "Disease Probability Distribution");
    autoTable(doc, {
      startY: y,
      margin: { left: 22, right: 22 },
      head: [["Condition", "Probability (%)", "Rank"]],
      body: result.probability_distribution.map((d, i) => [d.condition, `${d.probability}%`, `#${i + 1}`]),
      headStyles: { fillColor: COLORS.primary, fontSize: 8 },
      bodyStyles: { fontSize: 8, textColor: COLORS.text },
      alternateRowStyles: { fillColor: COLORS.bgLight },
    });
    y = (doc as any).lastAutoTable.finalY + 8;
  }

  // ── Clinical Reasoning ──
  if (result.clinical_reasoning) {
    y = checkPageBreak(doc, y, 30);
    y = addSectionHeader(doc, y, "Clinical Reasoning (AI Validation)");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.text);
    const reasonLines = doc.splitTextToSize(result.clinical_reasoning, pageWidth - 44);
    doc.text(reasonLines, 22, y);
    y += reasonLines.length * 4.5 + 8;
  }

  // ── Differential Diagnoses ──
  if (result.differential_diagnoses?.length) {
    y = checkPageBreak(doc, y, 30);
    y = addSectionHeader(doc, y, "Differential Diagnoses");
    autoTable(doc, {
      startY: y,
      margin: { left: 22, right: 22 },
      head: [["Condition", "Likelihood"]],
      body: result.differential_diagnoses.map((d) => [d.name, d.likelihood]),
      headStyles: { fillColor: COLORS.primary, fontSize: 8 },
      bodyStyles: { fontSize: 8, textColor: COLORS.text },
      alternateRowStyles: { fillColor: COLORS.bgLight },
    });
    y = (doc as any).lastAutoTable.finalY + 8;
  }

  // ── Recommendations ──
  if (result.recommendations?.length) {
    y = checkPageBreak(doc, y, 20 + result.recommendations.length * 6);
    y = addSectionHeader(doc, y, "Recommendations");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.text);
    result.recommendations.forEach((rec) => {
      y = checkPageBreak(doc, y, 8);
      doc.setFillColor(...COLORS.low);
      doc.circle(24, y - 1, 1.2, "F");
      const recLines = doc.splitTextToSize(rec, pageWidth - 50);
      doc.text(recLines, 28, y);
      y += recLines.length * 4.5 + 3;
    });
    y += 4;
  }

  // ── When to See a Doctor ──
  if (result.when_to_see_doctor) {
    y = checkPageBreak(doc, y, 25);
    y = addSectionHeader(doc, y, "When to See a Doctor");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(...COLORS.text);
    const doctorLines = doc.splitTextToSize(result.when_to_see_doctor, pageWidth - 44);
    doc.text(doctorLines, 22, y);
    y += doctorLines.length * 4.5 + 8;
  }

  // ── Preprocessing Steps ──
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
  y = checkPageBreak(doc, y, 28);
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.3);
  doc.line(14, y, pageWidth - 14, y);
  y += 6;
  doc.setFillColor(...COLORS.bgLight);
  doc.roundedRect(14, y, pageWidth - 28, 20, 2, 2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.muted);
  doc.text("DISCLAIMER", 18, y + 5);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(6.5);
  const disclaimer =
    "This report is generated by an ML model (XGBoost) validated by AI clinical reasoning. It is for informational and educational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for proper medical evaluation.";
  const disclaimerLines = doc.splitTextToSize(disclaimer, pageWidth - 36);
  doc.text(disclaimerLines, 18, y + 10);

  // ── Page numbers ──
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(...COLORS.muted);
    doc.text(`Page ${i} of ${totalPages}`, pageWidth / 2, 290, { align: "center" });
    doc.text("SinusAI Medical Report", 14, 290);
    doc.text(reportId, pageWidth - 14, 290, { align: "right" });
  }

  doc.save(`SinusAI-Report-${reportId}.pdf`);
}
