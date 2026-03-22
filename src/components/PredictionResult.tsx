import { AlertTriangle, CheckCircle2, Info, Shield, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PredictionData {
  condition: string;
  confidence: number;
  risk_level: "low" | "moderate" | "high";
  description: string;
  recommendations: string[];
  when_to_see_doctor: string;
  differential_diagnoses: { name: string; likelihood: string }[];
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
    ring: "bg-success/30",
  },
  moderate: {
    icon: Info,
    bg: "bg-warning/10",
    border: "border-warning/30",
    text: "text-warning",
    label: "Moderate Risk",
    ring: "bg-warning/30",
  },
  high: {
    icon: AlertTriangle,
    bg: "bg-destructive/10",
    border: "border-destructive/30",
    text: "text-destructive",
    label: "High Risk",
    ring: "bg-destructive/30",
  },
};

export function PredictionResult({ result, onReset }: PredictionResultProps) {
  const risk = riskConfig[result.risk_level];
  const RiskIcon = risk.icon;

  return (
    <div className="space-y-6">
      {/* Main prediction card */}
      <div className={`relative overflow-hidden rounded-2xl border-2 ${risk.border} ${risk.bg} p-6 opacity-0 animate-scale-in`}>
        <div className="absolute -top-8 -right-8 w-32 h-32 rounded-full opacity-20 animate-pulse-ring" style={{ background: `hsl(var(--risk-${result.risk_level}))` }} />
        <div className="relative">
          <div className="flex items-center gap-3 mb-4">
            <div className={`p-2 rounded-xl ${risk.bg} border ${risk.border}`}>
              <RiskIcon className={`w-6 h-6 ${risk.text}`} />
            </div>
            <div>
              <span className={`text-xs font-semibold uppercase tracking-wider ${risk.text}`}>{risk.label}</span>
              <p className="text-sm text-muted-foreground">{result.confidence}% confidence</p>
            </div>
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
            className={`h-full rounded-full transition-all duration-1000 ease-out`}
            style={{
              width: `${result.confidence}%`,
              background: `hsl(var(--risk-${result.risk_level}))`,
            }}
          />
        </div>
      </div>

      {/* Differential diagnoses */}
      {result.differential_diagnoses.length > 0 && (
        <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "250ms" }}>
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
      <div className="opacity-0 animate-fade-up rounded-xl border border-border bg-card p-5" style={{ animationDelay: "350ms" }}>
        <h3 className="font-display text-lg font-semibold text-foreground mb-3">Recommendations</h3>
        <ul className="space-y-2">
          {result.recommendations.map((rec, i) => (
            <li key={i} className="flex items-start gap-3 text-sm text-muted-foreground">
              <CheckCircle2 className="w-4 h-4 text-success mt-0.5 shrink-0" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* When to see doctor */}
      <div className="opacity-0 animate-fade-up rounded-xl border border-info/30 bg-info/5 p-5" style={{ animationDelay: "450ms" }}>
        <div className="flex items-start gap-3">
          <Shield className="w-5 h-5 text-info mt-0.5 shrink-0" />
          <div>
            <h3 className="font-display font-semibold text-foreground mb-1">When to See a Doctor</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{result.when_to_see_doctor}</p>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="opacity-0 animate-fade-up rounded-lg bg-muted/50 p-4" style={{ animationDelay: "550ms" }}>
        <p className="text-xs text-muted-foreground text-center leading-relaxed">
          <strong>Disclaimer:</strong> This prediction is generated by an AI model and is for informational purposes only.
          It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional
          for proper medical evaluation.
        </p>
      </div>

      <div className="opacity-0 animate-fade-up" style={{ animationDelay: "600ms" }}>
        <Button variant="outline" onClick={onReset} className="w-full h-12 rounded-xl active:scale-[0.98] transition-all">
          <ArrowLeft className="w-4 h-4" />
          New Assessment
        </Button>
      </div>
    </div>
  );
}
