import { AlertTriangle, CheckCircle2, Info, Shield, ArrowLeft, Brain, BarChart3, GitCompare, Layers, Lightbulb, Download, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { generateMedicalReport } from "@/utils/generateReport";

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface ModelComparison {
  accuracy: number;
  model: string;
}

interface ProbabilityDist {
  condition: string;
  probability: number;
}

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
    xgboost: ModelComparison;
    random_forest: ModelComparison;
    logistic_regression: ModelComparison;
    decision_tree: ModelComparison;
  };
  feature_importance: FeatureImportance[];
  probability_distribution: ProbabilityDist[];
  primary_model: string;
  pipeline_steps: string[];
  report_findings?: string;
  reports_analyzed?: number;
}

interface PredictionResultProps {
  result: PredictionData;
  onReset: () => void;
}

const riskConfig = {
  low: {
    icon: CheckCircle2,
    bg: "bg-success/10",
    border: "border-success/30",
    text: "text-success",
    label: "Low Risk",
  },
  moderate: {
    icon: Info,
    bg: "bg-warning/10",
    border: "border-warning/30",
    text: "text-warning",
    label: "Moderate Risk",
  },
  high: {
    icon: AlertTriangle,
    bg: "bg-destructive/10",
    border: "border-destructive/30",
    text: "text-destructive",
    label: "High Risk",
  },
};

const CHART_COLORS = ["hsl(187, 60%, 38%)", "hsl(160, 45%, 45%)", "hsl(205, 78%, 52%)", "hsl(38, 92%, 50%)", "hsl(0, 72%, 51%)"];

export function PredictionResult({ result, onReset }: PredictionResultProps) {
  const risk = riskConfig[result.risk_level];
  const RiskIcon = risk.icon;

  const models = result.model_comparison
    ? [
        result.model_comparison.xgboost,
        result.model_comparison.random_forest,
        result.model_comparison.logistic_regression,
        result.model_comparison.decision_tree,
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Pipeline Steps */}
      {result.pipeline_steps && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="w-4 h-4 text-primary" />
            <h3 className="font-display text-sm font-semibold text-foreground">ML Pipeline Executed</h3>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.pipeline_steps.map((step, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded-md bg-primary/10 text-primary">
                <span className="w-3.5 h-3.5 rounded-full bg-primary/20 flex items-center justify-center text-[9px] font-bold">{i + 1}</span>
                {step}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Main prediction card */}
      <div className={`relative overflow-hidden rounded-2xl border-2 ${risk.border} ${risk.bg} p-6 opacity-0 animate-scale-in`}>
        <div className="absolute -top-8 -right-8 w-32 h-32 rounded-full opacity-20 animate-pulse-ring" style={{ background: `hsl(var(--risk-${result.risk_level}))` }} />
        <div className="relative">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl ${risk.bg} border ${risk.border}`}>
                <RiskIcon className={`w-6 h-6 ${risk.text}`} />
              </div>
              <div>
                <span className={`text-xs font-semibold uppercase tracking-wider ${risk.text}`}>{risk.label}</span>
                <p className="text-sm text-muted-foreground">{result.confidence}% confidence</p>
              </div>
            </div>
            {result.primary_model && (
              <span className="text-[10px] font-medium px-2 py-1 rounded-md bg-card/60 text-muted-foreground border border-border">
                {result.primary_model}
              </span>
            )}
          </div>
          <h2 className="font-display text-2xl font-bold text-foreground mb-2">{result.condition}</h2>
          <p className="text-muted-foreground leading-relaxed">{result.description}</p>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="opacity-0 animate-fade-up" style={{ animationDelay: "150ms" }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-foreground">Prediction Confidence</span>
          <span className={`text-sm font-bold ${risk.text}`}>{result.confidence}%</span>
        </div>
        <div className="h-3 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${result.confidence}%`,
              background: `hsl(var(--risk-${result.risk_level}))`,
            }}
          />
        </div>
      </div>

      {/* Model Comparison */}
      {models.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "200ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <GitCompare className="w-4 h-4 text-primary" />
            <h3 className="font-display text-lg font-semibold text-foreground">Model Comparison</h3>
          </div>
          <div className="space-y-3">
            {models.map((m, i) => {
              const isTop = i === 0;
              return (
                <div key={m.model} className="flex items-center gap-3">
                  <span className={`text-xs font-medium w-32 shrink-0 ${isTop ? "text-primary font-semibold" : "text-muted-foreground"}`}>
                    {m.model} {isTop && "★"}
                  </span>
                  <div className="flex-1 h-6 rounded-md bg-muted overflow-hidden relative">
                    <div
                      className="h-full rounded-md transition-all duration-700 ease-out"
                      style={{
                        width: `${m.accuracy}%`,
                        background: CHART_COLORS[i],
                        animationDelay: `${i * 100}ms`,
                      }}
                    />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-bold text-foreground">
                      {m.accuracy}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Feature Importance Chart */}
      {result.feature_importance && result.feature_importance.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "300ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-4 h-4 text-primary" />
            <h3 className="font-display text-lg font-semibold text-foreground">Feature Importance (Explainability)</h3>
          </div>
          <ResponsiveContainer width="100%" height={Math.max(160, result.feature_importance.length * 36)}>
            <BarChart data={result.feature_importance} layout="vertical" margin={{ left: 20, right: 20, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(200, 18%, 88%)" />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: "hsl(210, 10%, 46%)" }} />
              <YAxis dataKey="feature" type="category" width={120} tick={{ fontSize: 11, fill: "hsl(210, 25%, 12%)" }} />
              <Tooltip
                contentStyle={{
                  background: "hsl(0, 0%, 100%)",
                  border: "1px solid hsl(200, 18%, 88%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {result.feature_importance.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Probability Distribution */}
      {result.probability_distribution && result.probability_distribution.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "350ms" }}>
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-primary" />
            <h3 className="font-display text-lg font-semibold text-foreground">Disease Probability Distribution</h3>
          </div>
          <div className="space-y-2">
            {result.probability_distribution.map((d, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className={`text-xs font-medium w-36 shrink-0 ${i === 0 ? "text-primary font-semibold" : "text-muted-foreground"}`}>
                  {d.condition}
                </span>
                <div className="flex-1 h-5 rounded-md bg-muted overflow-hidden relative">
                  <div
                    className="h-full rounded-md"
                    style={{ width: `${d.probability}%`, background: CHART_COLORS[i % CHART_COLORS.length] }}
                  />
                </div>
                <span className="text-xs font-bold text-foreground w-12 text-right">{d.probability}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clinical Reasoning */}
      {result.clinical_reasoning && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-primary/20 bg-primary/5 p-5" style={{ animationDelay: "400ms" }}>
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-primary mt-0.5 shrink-0" />
            <div>
              <h3 className="font-display font-semibold text-foreground mb-1">Clinical Reasoning (AI Validation)</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{result.clinical_reasoning}</p>
            </div>
          </div>
        </div>
      )}

      {/* Report Findings */}
      {result.report_findings && result.report_findings.trim() !== "" && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-info/30 bg-info/5 p-5" style={{ animationDelay: "420ms" }}>
          <div className="flex items-start gap-3">
            <FileText className="w-5 h-5 text-info mt-0.5 shrink-0" />
            <div>
              <h3 className="font-display font-semibold text-foreground mb-1">
                Medical Report Findings ({result.reports_analyzed} report{result.reports_analyzed !== 1 ? "s" : ""} analyzed)
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">{result.report_findings}</p>
            </div>
          </div>
        </div>
      )}


      {result.differential_diagnoses?.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "450ms" }}>
          <h3 className="font-display text-lg font-semibold text-foreground mb-3">Differential Diagnoses</h3>
          <div className="space-y-2">
            {result.differential_diagnoses.map((d, i) => (
              <div key={i} className="flex items-center justify-between py-2 px-3 rounded-lg bg-surface-sunken">
                <span className="text-sm font-medium text-foreground">{d.name}</span>
                <span className="text-xs font-medium text-muted-foreground px-2 py-1 rounded-md bg-muted">{d.likelihood}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "500ms" }}>
        <h3 className="font-display text-lg font-semibold text-foreground mb-3">Recommendations</h3>
        <ul className="space-y-2">
          {result.recommendations?.map((rec, i) => (
            <li key={i} className="flex items-start gap-3 text-sm text-muted-foreground">
              <CheckCircle2 className="w-4 h-4 text-success mt-0.5 shrink-0" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Preprocessing Steps */}
      {result.preprocessing_steps && result.preprocessing_steps.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "550ms" }}>
          <h3 className="font-display text-sm font-semibold text-foreground mb-3">Data Preprocessing Applied</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {result.preprocessing_steps.map((step, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
                {step}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* When to see doctor */}
      <div className="opacity-0 animate-fade-up rounded-xl border border-info/30 bg-info/5 p-5" style={{ animationDelay: "600ms" }}>
        <div className="flex items-start gap-3">
          <Shield className="w-5 h-5 text-info mt-0.5 shrink-0" />
          <div>
            <h3 className="font-display font-semibold text-foreground mb-1">When to See a Doctor</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{result.when_to_see_doctor}</p>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="opacity-0 animate-fade-up rounded-lg bg-muted/50 p-4" style={{ animationDelay: "650ms" }}>
        <p className="text-xs text-muted-foreground text-center leading-relaxed">
          <strong>Disclaimer:</strong> This prediction is generated by an ML model (XGBoost) validated by AI clinical reasoning.
          It is for informational and educational purposes only. It does not constitute medical advice, diagnosis, or treatment.
          Always consult a qualified healthcare professional for proper medical evaluation.
        </p>
      </div>

      <div className="opacity-0 animate-fade-up flex gap-3" style={{ animationDelay: "700ms" }}>
        <Button
          onClick={() => generateMedicalReport(result)}
          className="flex-1 h-12 rounded-xl active:scale-[0.98] transition-all"
        >
          <Download className="w-4 h-4" />
          Download PDF Report
        </Button>
        <Button variant="outline" onClick={onReset} className="flex-1 h-12 rounded-xl active:scale-[0.98] transition-all">
          <ArrowLeft className="w-4 h-4" />
          New Assessment
        </Button>
      </div>
    </div>
  );
}
