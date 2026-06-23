import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  Gauge,
  Network,
  Waves,
} from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type FileInfo = {
  path?: string | null;
  exists?: boolean;
  url?: string | null;
};

type MotorMetrics = {
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  best_subject?: number | string | null;
  num_subjects?: number | string | unknown[] | null;
  num_channels?: number | string | unknown[] | null;
  num_csp_components?: number | string | unknown[] | null;
  left_hand_count?: number | string | null;
  right_hand_count?: number | string | null;
};

type MotorStatus = {
  module?: string;
  status?: string;
  metrics?: MotorMetrics | null;
  outputs?: {
    confusion_matrix?: FileInfo;
    subject_comparison_plot?: FileInfo;
    csp_topomap?: FileInfo;
    csp_3d_topomap?: FileInfo;
    csp_3d_topomap_metadata?: FileInfo;
    predictions_csv?: FileInfo;
    metrics_json?: FileInfo;
    subject_comparison_csv?: FileInfo;
    subject_comparison_best_json?: FileInfo;
    model_file?: FileInfo;
  };
};

export default async function MotorPage() {
  const status = await safeApiFetch<MotorStatus>("/api/motor/status");
  const metrics = status?.metrics ?? null;
  const ready = status?.status === "ready";

  const primaryScore = metrics?.accuracy ?? metrics?.f1_score ?? null;

  const csp3dUrl =
    status?.outputs?.csp_3d_topomap?.url
      ? mediaUrl(status.outputs.csp_3d_topomap.url)
      : mediaUrl("motor_imagery/motor_imagery_csp_3d_topomap.html");

  const csp3dAvailable =
    status?.outputs?.csp_3d_topomap?.exists ?? false;

  return (
    <AppShell
      activePath="/motor"
      title="EEG Motor Imagery BCI"
      subtitle="CSP + LDA motor imagery classification dashboard with subject comparison, 2D CSP topomap, and interactive 3D CSP scalp visualization"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Gauge}
          label="Accuracy"
          value={formatMetric(metrics?.accuracy)}
          description="Motor imagery classification accuracy"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={Activity}
          label="F1 Score"
          value={formatMetric(metrics?.f1_score)}
          description="Classification balance"
          iconClassName="bg-purple-50 text-purple-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="Best Subject"
          value={formatSafeCount(metrics?.best_subject)}
          description="Best PhysioNet subject"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Network}
          label="CSP Components"
          value={formatSafeCount(metrics?.num_csp_components)}
          description="Spatial filtering components"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Motor imagery outputs
            </h2>

            <p className="mt-1 max-w-3xl text-sm text-slate-500">
              This section shows classification performance, subject-level
              accuracy comparison, and the standard 2D CSP spatial topomap.
            </p>
          </div>

          <StatusBadge active={ready} />
        </div>

        <div className="mt-5 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          <ImagePanel
            title="Confusion Matrix"
            description="Left-hand vs right-hand classification results"
            src={mediaUrl("motor_imagery/motor_imagery_confusion_matrix.png")}
            alt="Motor imagery confusion matrix"
          />

          <ImagePanel
            title="Subject Comparison"
            description="Accuracy comparison across PhysioNet subjects"
            src={mediaUrl(
              "motor_imagery/physionet_subject_comparison_accuracy.png",
            )}
            alt="PhysioNet subject comparison"
          />

          <ImagePanel
            title="CSP Topomap 2D"
            description="Standard 2D CSP spatial pattern"
            src={mediaUrl("motor_imagery/motor_imagery_csp_topomap.png")}
            alt="Motor imagery CSP 2D topomap"
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              CSP 3D Scalp Visualization
            </h2>

            <p className="mt-1 max-w-3xl text-sm text-slate-500">
              Interactive 3D visualization of CSP spatial pattern weights mapped
              onto approximate EEG electrode positions.
            </p>
          </div>

          <StatusBadge active={csp3dAvailable} />
        </div>

        <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-slate-950">
          <iframe
            src={csp3dUrl}
            title="Motor imagery CSP 3D scalp visualization"
            className="h-[620px] w-full"
          />
        </div>

        <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          This 3D CSP map is a model-derived spatial pattern, not a direct brain
          activation map. Drag to rotate, scroll to zoom, and hover over
          electrodes to inspect CSP weights.
        </p>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Classification metrics"
          icon={BarChart3}
          values={[
            ["Primary score", formatMetric(primaryScore)],
            ["Accuracy", formatMetric(metrics?.accuracy)],
            ["Precision", formatMetric(metrics?.precision)],
            ["Recall", formatMetric(metrics?.recall)],
            ["F1 score", formatMetric(metrics?.f1_score)],
            ["Best subject", formatSafeCount(metrics?.best_subject)],
          ]}
        />

        <InfoPanel
          title="EEG dataset summary"
          icon={Database}
          values={[
            ["Number of subjects", formatSafeCount(metrics?.num_subjects)],
            ["EEG channels", formatSafeCount(metrics?.num_channels)],
            [
              "CSP components",
              formatSafeCount(metrics?.num_csp_components),
            ],
            ["Left-hand samples", formatSafeCount(metrics?.left_hand_count)],
            ["Right-hand samples", formatSafeCount(metrics?.right_hand_count)],
          ]}
        />

        <InfoPanel
          title="Output files"
          icon={FileText}
          values={[
            [
              "Confusion matrix",
              fileStatus(status?.outputs?.confusion_matrix),
            ],
            [
              "Subject comparison plot",
              fileStatus(status?.outputs?.subject_comparison_plot),
            ],
            ["2D CSP topomap", fileStatus(status?.outputs?.csp_topomap)],
            [
              "3D CSP topomap",
              fileStatus(status?.outputs?.csp_3d_topomap),
            ],
            [
              "3D CSP metadata",
              fileStatus(status?.outputs?.csp_3d_topomap_metadata),
            ],
            ["Predictions CSV", fileStatus(status?.outputs?.predictions_csv)],
            ["Metrics JSON", fileStatus(status?.outputs?.metrics_json)],
            ["Model file", fileStatus(status?.outputs?.model_file)],
          ]}
        />

        <InfoPanel
          title="Pipeline"
          icon={Waves}
          values={[
            ["Step 1", "Load PhysioNet EEGBCI data"],
            ["Step 2", "Filter EEG and extract epochs"],
            ["Step 3", "Apply CSP spatial filtering"],
            ["Step 4", "Classify with LDA"],
            ["Step 5", "Generate 2D and 3D CSP visualizations"],
          ]}
        />

        <InfoPanel
          title="Model details"
          icon={BrainCircuit}
          values={[
            ["Model", "CSP + LDA"],
            ["Input", "Motor imagery EEG epochs"],
            ["Task", "Left-hand vs right-hand imagery"],
            ["2D visualization", "CSP topomap"],
            ["3D visualization", "Interactive CSP scalp map"],
            ["Module", status?.module ?? "EEG motor imagery BCI"],
            ["Status", status?.status ?? "Unavailable"],
          ]}
        />
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        The CSP topomap is a model-derived spatial pattern, not a direct brain
        activation map. This module is for research and portfolio demonstration
        only.
      </section>
    </AppShell>
  );
}

function ImagePanel({
  title,
  description,
  src,
  alt,
}: {
  title: string;
  description?: string;
  src: string;
  alt: string;
}) {
  return (
    <article className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 p-4">
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>

        {description ? (
          <p className="mt-1 text-xs text-slate-500">{description}</p>
        ) : null}
      </div>

      <div className="flex h-96 items-center justify-center bg-slate-950 p-3">
        <img
          src={src}
          alt={alt}
          className="max-h-full max-w-full object-contain"
        />
      </div>
    </article>
  );
}

function InfoPanel({
  title,
  icon: Icon,
  values,
}: {
  title: string;
  icon: React.ElementType;
  values: Array<[string, string]>;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-blue-50 p-2 text-blue-600">
          <Icon className="h-5 w-5" />
        </div>

        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
      </div>

      <dl className="mt-4 divide-y divide-slate-100">
        {values.map(([label, value]) => (
          <MetricRow key={label} label={label} value={value} />
        ))}
      </dl>
    </article>
  );
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <dt className="text-sm text-slate-500">{label}</dt>

      <dd className="text-right text-sm font-semibold text-slate-800">
        {value}
      </dd>
    </div>
  );
}

async function safeApiFetch<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch {
    return null;
  }
}

function formatMetric(value?: number | null): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatSafeCount(
  value?: number | string | unknown[] | null,
): string {
  if (value == null) return "N/A";
  if (Array.isArray(value)) return value.length.toLocaleString();
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string") return value;
  return "N/A";
}

function fileStatus(file?: FileInfo): string {
  return file?.exists ? "Available" : "Missing";
}