import Link from "next/link";
import type { ElementType } from "react";

import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type FileInfo = {
  path?: string | null;
  exists: boolean;
  url?: string | null;
};

type MriMetrics = {
  case_mean_dice?: number;
  case_mean_iou?: number;
  slice_mean_dice?: number;
  slice_mean_iou?: number;
  num_cases?: number | string | unknown[] | null;
  num_slices?: number | string | unknown[] | null;
};

type MriStatus = {
  module?: string;
  status?: string;
  metrics?: MriMetrics | null;
  outputs?: {
    prediction_overlay?: FileInfo;
    best_case_overlay?: FileInfo;
    worst_case_overlay?: FileInfo;
  };
  best_worst_cases?: {
    available?: boolean;
    best_case_overlay?: FileInfo;
    worst_case_overlay?: FileInfo;
  };
};

type SeizureMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  sensitivity?: number;
  specificity?: number;
  f1_score?: number;
  balanced_accuracy?: number;
  num_files?: number;
  num_windows?: number;
  num_samples?: number;
};

type SeizureStatus = {
  module?: string;
  status?: string;
  metrics?: SeizureMetrics | null;
  multi_file_metrics?: SeizureMetrics | null;
  outputs?: {
    confusion_matrix?: FileInfo;
    multi_file_confusion_matrix?: FileInfo;
    waveform?: FileInfo;
    probability_timeline?: FileInfo;
  };
  visualization?: {
    available?: boolean;
  };
};

type MotorMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  best_subject?: number | string;
  num_subjects?: number;
  num_channels?: number;
  num_csp_components?: number;
};

type MotorStatus = {
  module?: string;
  status?: string;
  metrics?: MotorMetrics | null;
  outputs?: {
    confusion_matrix?: FileInfo;
    subject_comparison_plot?: FileInfo;
    csp_topomap?: FileInfo;
  };
};

type ModuleCardProps = {
  title: string;
  description: string;
  href: string;
  icon: ElementType;
  status?: string;
  scoreLabel: string;
  scoreValue: string;
  previewSrc: string;
  details: Array<[string, string]>;
};

export default async function OverviewPage() {
  const [mriStatus, seizureStatus, motorStatus] = await Promise.all([
    safeApiFetch<MriStatus>("/api/mri/status"),
    safeApiFetch<SeizureStatus>("/api/seizure/status"),
    safeApiFetch<MotorStatus>("/api/motor/status"),
  ]);

  const mriMetrics = mriStatus?.metrics ?? null;

  const seizureMetrics =
    seizureStatus?.multi_file_metrics ??
    seizureStatus?.metrics ??
    null;

  const motorMetrics = motorStatus?.metrics ?? null;

  const mriScore =
    mriMetrics?.case_mean_dice ??
    mriMetrics?.slice_mean_dice;

  const seizureScore =
    seizureMetrics?.f1_score ??
    seizureMetrics?.balanced_accuracy ??
    seizureMetrics?.accuracy;

  const motorScore =
    motorMetrics?.accuracy ??
    motorMetrics?.f1_score;

  const mriPreview =
    mriStatus?.best_worst_cases?.best_case_overlay?.url
      ? mediaUrl(mriStatus.best_worst_cases.best_case_overlay.url)
      : mriStatus?.outputs?.prediction_overlay?.url
        ? mediaUrl(mriStatus.outputs.prediction_overlay.url)
        : mediaUrl("mri/mri_prediction_overlay.png");

  const seizurePreview =
    seizureStatus?.outputs?.probability_timeline?.url
      ? mediaUrl(seizureStatus.outputs.probability_timeline.url)
      : seizureStatus?.outputs?.waveform?.url
        ? mediaUrl(seizureStatus.outputs.waveform.url)
        : seizureStatus?.outputs?.multi_file_confusion_matrix?.url
          ? mediaUrl(seizureStatus.outputs.multi_file_confusion_matrix.url)
          : seizureStatus?.outputs?.confusion_matrix?.url
            ? mediaUrl(seizureStatus.outputs.confusion_matrix.url)
            : mediaUrl("seizure/seizure_probability_timeline.png");

  const motorPreview =
    motorStatus?.outputs?.csp_topomap?.url
      ? mediaUrl(motorStatus.outputs.csp_topomap.url)
      : motorStatus?.outputs?.subject_comparison_plot?.url
        ? mediaUrl(motorStatus.outputs.subject_comparison_plot.url)
        : motorStatus?.outputs?.confusion_matrix?.url
          ? mediaUrl(motorStatus.outputs.confusion_matrix.url)
          : mediaUrl("motor_imagery/motor_imagery_csp_topomap.png");

  const readyCount = [
    mriStatus?.status,
    seizureStatus?.status,
    motorStatus?.status,
  ].filter((status) => status === "ready").length;

  return (
    <AppShell
      activePath="/"
      title="NeuroFusion-AI"
      subtitle="Unified neuroscience AI dashboard for MRI segmentation, EEG seizure detection, and motor imagery BCI"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={BrainCircuit}
          label="Active Modules"
          value="3"
          description="MRI, seizure, motor imagery"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={ShieldCheck}
          label="Ready APIs"
          value={`${readyCount}/3`}
          description="Live backend status"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Gauge}
          label="MRI Dice"
          value={formatMetric(mriScore)}
          description="Case or slice Dice"
          iconClassName="bg-purple-50 text-purple-600"
        />

        <StatCard
          icon={Activity}
          label="Seizure F1"
          value={formatMetric(seizureScore)}
          description="Detection performance"
          iconClassName="bg-rose-50 text-rose-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Project overview
            </h2>

            <p className="mt-1 max-w-3xl text-sm text-slate-500">
              NeuroFusion-AI connects multiple neuroscience AI workflows into
              one full-stack dashboard. This overview now reads live metrics
              and result images from the FastAPI backend.
            </p>
          </div>

          <StatusBadge active={readyCount > 0} />
        </div>

        <div className="mt-5 grid gap-5 xl:grid-cols-3">
          <ModuleCard
            title="MRI Tumor Segmentation"
            description="2D U-Net style brain tumor segmentation with Dice, IoU, overlay, and best/worst validation case review."
            href="/mri"
            icon={ScanLine}
            status={mriStatus?.status}
            scoreLabel="Dice"
            scoreValue={formatMetric(mriScore)}
            previewSrc={mriPreview}
            details={[
              ["Case Dice", formatMetric(mriMetrics?.case_mean_dice)],
              ["Case IoU", formatMetric(mriMetrics?.case_mean_iou)],
              ["Slice Dice", formatMetric(mriMetrics?.slice_mean_dice)],
              ["Cases", formatSafeCount(mriMetrics?.num_cases)],
            ]}
          />

          <ModuleCard
            title="EEG Seizure Detection"
            description="EEG seizure classification with Random Forest metrics, confusion matrix, waveform preview, and probability timeline."
            href="/seizure"
            icon={Activity}
            status={seizureStatus?.status}
            scoreLabel="F1 / Accuracy"
            scoreValue={formatMetric(seizureScore)}
            previewSrc={seizurePreview}
            details={[
              ["Accuracy", formatMetric(seizureMetrics?.accuracy)],
              ["F1 score", formatMetric(seizureMetrics?.f1_score)],
              [
                "Sensitivity",
                formatMetric(
                  seizureMetrics?.sensitivity ??
                    seizureMetrics?.recall,
                ),
              ],
              [
                "Windows",
                formatSafeCount(
                  seizureMetrics?.num_windows ??
                    seizureMetrics?.num_samples,
                ),
              ],
            ]}
          />

          <ModuleCard
            title="EEG Motor Imagery BCI"
            description="CSP + LDA motor imagery classification with PhysioNet subject search and CSP spatial topomap visualization."
            href="/motor"
            icon={BrainCircuit}
            status={motorStatus?.status}
            scoreLabel="Accuracy"
            scoreValue={formatMetric(motorScore)}
            previewSrc={motorPreview}
            details={[
              ["Accuracy", formatMetric(motorMetrics?.accuracy)],
              ["F1 score", formatMetric(motorMetrics?.f1_score)],
              [
                "Best subject",
                formatSafeCount(motorMetrics?.best_subject),
              ],
              [
                "CSP components",
                formatSafeCount(motorMetrics?.num_csp_components),
              ],
            ]}
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Live API status"
          icon={Database}
          values={[
            ["MRI API", mriStatus?.status ?? "Unavailable"],
            ["Seizure API", seizureStatus?.status ?? "Unavailable"],
            ["Motor API", motorStatus?.status ?? "Unavailable"],
            ["Backend source", "FastAPI status endpoints"],
          ]}
        />

        <InfoPanel
          title="Current model summary"
          icon={BarChart3}
          values={[
            ["MRI score", formatMetric(mriScore)],
            ["Seizure score", formatMetric(seizureScore)],
            ["Motor score", formatMetric(motorScore)],
            ["Ready modules", `${readyCount}/3`],
          ]}
        />
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        This dashboard is for research, education, and portfolio demonstration
        only. It is not intended for clinical diagnosis or real-time patient
        monitoring.
      </section>
    </AppShell>
  );
}

async function safeApiFetch<T>(path: string): Promise<T | null> {
  try {
    return await apiFetch<T>(path);
  } catch {
    return null;
  }
}

function ModuleCard({
  title,
  description,
  href,
  icon: Icon,
  status,
  scoreLabel,
  scoreValue,
  previewSrc,
  details,
}: ModuleCardProps) {
  return (
    <article className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-start justify-between gap-4 p-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-blue-50 p-2 text-blue-600">
            <Icon className="h-5 w-5" />
          </div>

          <div>
            <h3 className="text-base font-semibold text-slate-900">
              {title}
            </h3>

            <p className="mt-1 text-xs text-slate-500">
              {scoreLabel}:{" "}
              <span className="font-semibold text-slate-800">
                {scoreValue}
              </span>
            </p>
          </div>
        </div>

        <StatusBadge active={status === "ready"} />
      </div>

      <div className="flex h-56 items-center justify-center bg-slate-950 p-3">
        <img
          src={previewSrc}
          alt={`${title} preview`}
          className="max-h-full max-w-full object-contain"
        />
      </div>

      <div className="p-5">
        <p className="text-sm leading-6 text-slate-500">
          {description}
        </p>

        <dl className="mt-4 divide-y divide-slate-100">
          {details.map(([label, value]) => (
            <MetricRow
              key={label}
              label={label}
              value={value}
            />
          ))}
        </dl>

        <Link
          href={href}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700"
        >
          Open module
          <ArrowRight className="h-4 w-4" />
        </Link>
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
  icon: ElementType;
  values: Array<[string, string]>;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-blue-50 p-2 text-blue-600">
          <Icon className="h-5 w-5" />
        </div>

        <h2 className="text-lg font-semibold text-slate-900">
          {title}
        </h2>
      </div>

      <dl className="mt-4 divide-y divide-slate-100">
        {values.map(([label, value]) => (
          <MetricRow
            key={label}
            label={label}
            value={value}
          />
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

function formatMetric(value?: number | null): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatSafeCount(
  value?: number | string | unknown[] | null,
): string {
  if (value == null) {
    return "N/A";
  }

  if (Array.isArray(value)) {
    return value.length.toLocaleString();
  }

  if (typeof value === "number") {
    return value.toLocaleString();
  }

  if (typeof value === "string") {
    return value;
  }

  return "N/A";
}