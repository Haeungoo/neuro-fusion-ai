import {
  BrainCircuit,
  Database,
  Gauge,
  ScanLine,
  TrendingDown,
  TrendingUp,
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
  model?: string;
  dataset?: string;
};

type BestWorstCase = {
  case_id?: string;
  dice?: number | null;
  output_image?: string;
  source?: Record<string, unknown>;
};

type BestWorstMetadata = {
  mode?: string;
  best_case?: BestWorstCase;
  worst_case?: BestWorstCase;
  note?: string;
};

type MriStatus = {
  module?: string;
  status?: string;
  description?: string;
  dataset_note?: string;

  metrics?: MriMetrics | null;

  model?: {
    name?: string;
    task?: string;
    input?: string;
    output?: string;
    model_file?: FileInfo;
  };

  outputs?: {
    metrics_json?: FileInfo;
    input_slice?: FileInfo;
    ground_truth_mask?: FileInfo;
    predicted_mask?: FileInfo;
    prediction_overlay?: FileInfo;
    best_case_overlay?: FileInfo;
    worst_case_overlay?: FileInfo;
    best_worst_metadata?: FileInfo;
    model_file?: FileInfo;
  };

  visualization?: {
    available?: boolean;
    input_slice?: FileInfo;
    ground_truth_mask?: FileInfo;
    predicted_mask?: FileInfo;
    prediction_overlay?: FileInfo;
  };

  best_worst_cases?: {
    available?: boolean;
    metadata?: BestWorstMetadata | null;
    metadata_file?: FileInfo;
    best_case_overlay?: FileInfo;
    worst_case_overlay?: FileInfo;
  };

  pipeline?: string[];
  disclaimer?: string;
};

export default async function MriPage() {
  const status = await apiFetch<MriStatus>("/api/mri/status");

  const metrics = status?.metrics ?? null;

  const inputSliceUrl =
    status?.outputs?.input_slice?.url
      ? mediaUrl(status.outputs.input_slice.url)
      : mediaUrl("mri/mri_input_slice.png");

  const groundTruthUrl =
    status?.outputs?.ground_truth_mask?.url
      ? mediaUrl(status.outputs.ground_truth_mask.url)
      : mediaUrl("mri/mri_ground_truth_mask.png");

  const predictedMaskUrl =
    status?.outputs?.predicted_mask?.url
      ? mediaUrl(status.outputs.predicted_mask.url)
      : mediaUrl("mri/mri_predicted_mask.png");

  const overlayUrl =
    status?.outputs?.prediction_overlay?.url
      ? mediaUrl(status.outputs.prediction_overlay.url)
      : mediaUrl("mri/mri_prediction_overlay.png");

  const bestWorstMetadata = status?.best_worst_cases?.metadata ?? null;

  const bestCase = bestWorstMetadata?.best_case ?? null;
  const worstCase = bestWorstMetadata?.worst_case ?? null;

  const bestWorstAvailable =
    status?.best_worst_cases?.available ??
    Boolean(
      status?.outputs?.best_case_overlay?.exists &&
        status?.outputs?.worst_case_overlay?.exists,
    );

  const modelFile =
    status?.outputs?.model_file ??
    status?.model?.model_file;

  return (
    <AppShell
      activePath="/mri"
      title="MRI Tumor Segmentation"
      subtitle="Brain tumor segmentation using a 2D U-Net style workflow"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard
          icon={Gauge}
          label="Case Dice"
          value={formatMetric(metrics?.case_mean_dice)}
          description="Mean Dice by case"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={ScanLine}
          label="Case IoU"
          value={formatMetric(metrics?.case_mean_iou)}
          description="Mean IoU by case"
          iconClassName="bg-teal-50 text-teal-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="Slice Dice"
          value={formatMetric(metrics?.slice_mean_dice)}
          description="Mean Dice by slice"
          iconClassName="bg-purple-50 text-purple-600"
        />

        <StatCard
          icon={Database}
          label="Slice IoU"
          value={formatMetric(metrics?.slice_mean_iou)}
          description="Mean IoU by slice"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Database}
          label="Cases"
          value={formatSafeCount(metrics?.num_cases)}
          description="Validation cases"
          iconClassName="bg-cyan-50 text-cyan-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="Status"
          value={status?.status ?? "N/A"}
          description="Backend output check"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              MRI segmentation outputs
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Input MRI, ground-truth mask, predicted tumor mask, and final
              segmentation overlay.
            </p>
          </div>

          <StatusBadge active={status?.visualization?.available ?? false} />
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
          <ImagePanel
            title="Input MRI"
            src={inputSliceUrl}
            compact
          />

          <ImagePanel
            title="Ground Truth Mask"
            src={groundTruthUrl}
            compact
          />

          <ImagePanel
            title="Predicted Mask"
            src={predictedMaskUrl}
            compact
          />

          <ImagePanel
            title="Prediction Overlay"
            src={overlayUrl}
            compact
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Best Validation Case
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              The best validation case is shown using the same four-panel
              layout as the MRI segmentation outputs.
            </p>
          </div>

          <StatusBadge active={bestWorstAvailable} />
        </div>

        <div className="mt-4 rounded-xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-medium text-blue-800">
          {`Best case${
            bestCase?.case_id ? ` · ${bestCase.case_id}` : ""
          }${
            bestCase?.dice != null
              ? ` · Dice ${bestCase.dice.toFixed(4)}`
              : ""
          }`}
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
          <ImagePanel
            title="Best Case Input MRI"
            src={mediaUrl("mri/mri_best_case_input.png")}
            compact
          />

          <ImagePanel
            title="Best Case Ground Truth"
            src={mediaUrl("mri/mri_best_case_ground_truth.png")}
            compact
          />

          <ImagePanel
            title="Best Case Predicted Mask"
            src={mediaUrl("mri/mri_best_case_predicted_mask.png")}
            compact
          />

          <ImagePanel
            title="Best Case Overlay"
            src={mediaUrl("mri/mri_best_case_overlay_only.png")}
            compact
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Worst Validation Case
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              The worst validation case is shown in the same four-panel format
              to help identify where the segmentation model struggles.
            </p>
          </div>

          <StatusBadge active={bestWorstAvailable} />
        </div>

        <div className="mt-4 rounded-xl border border-rose-100 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-800">
          {`Worst case${
            worstCase?.case_id ? ` · ${worstCase.case_id}` : ""
          }${
            worstCase?.dice != null
              ? ` · Dice ${worstCase.dice.toFixed(4)}`
              : ""
          }`}
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-2 xl:grid-cols-4">
          <ImagePanel
            title="Worst Case Input MRI"
            src={mediaUrl("mri/mri_worst_case_input.png")}
            compact
          />

          <ImagePanel
            title="Worst Case Ground Truth"
            src={mediaUrl("mri/mri_worst_case_ground_truth.png")}
            compact
          />

          <ImagePanel
            title="Worst Case Predicted Mask"
            src={mediaUrl("mri/mri_worst_case_predicted_mask.png")}
            compact
          />

          <ImagePanel
            title="Worst Case Overlay"
            src={mediaUrl("mri/mri_worst_case_overlay_only.png")}
            compact
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Best case details"
          icon={TrendingUp}
          values={[
            ["Case ID", bestCase?.case_id ?? "N/A"],
            ["Dice score", formatMetricOrNA(bestCase?.dice)],
            ["Mode", bestWorstMetadata?.mode ?? "N/A"],
            [
              "Metadata file",
              status?.best_worst_cases?.metadata_file?.exists ||
              status?.outputs?.best_worst_metadata?.exists
                ? "Saved"
                : "Missing",
            ],
          ]}
        />

        <InfoPanel
          title="Worst case details"
          icon={TrendingDown}
          values={[
            ["Case ID", worstCase?.case_id ?? "N/A"],
            ["Dice score", formatMetricOrNA(worstCase?.dice)],
            ["Mode", bestWorstMetadata?.mode ?? "N/A"],
            [
              "Best/Worst status",
              bestWorstAvailable ? "Available" : "Missing",
            ],
          ]}
        />
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Model details"
          icon={BrainCircuit}
          values={[
            ["Model", status?.model?.name ?? metrics?.model ?? "2D U-Net"],
            ["Task", status?.model?.task ?? "Brain tumor segmentation"],
            ["Dataset", metrics?.dataset ?? "BraTS-style MRI slices"],
            ["Input", status?.model?.input ?? "MRI slice"],
            ["Output", status?.model?.output ?? "Tumor mask"],
            ["Model file", modelFile?.exists ? "Saved" : "Missing"],
          ]}
        />

        <InfoPanel
          title="Output files"
          icon={Database}
          values={[
            [
              "Metrics JSON",
              status?.outputs?.metrics_json?.exists ? "Saved" : "Missing",
            ],
            [
              "Input slice",
              status?.outputs?.input_slice?.exists ? "Saved" : "Missing",
            ],
            [
              "Ground truth mask",
              status?.outputs?.ground_truth_mask?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Predicted mask",
              status?.outputs?.predicted_mask?.exists ? "Saved" : "Missing",
            ],
            [
              "Prediction overlay",
              status?.outputs?.prediction_overlay?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Best/Worst metadata",
              status?.outputs?.best_worst_metadata?.exists ||
              status?.best_worst_cases?.metadata_file?.exists
                ? "Saved"
                : "Missing",
            ],
          ]}
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">
          MRI segmentation pipeline
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          The pipeline converts MRI slices into tumor segmentation masks and
          validation overlays.
        </p>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-7">
          {(status?.pipeline ?? [
            "MRI slice loading",
            "Preprocessing",
            "U-Net segmentation",
            "Prediction mask generation",
            "Dice and IoU evaluation",
            "Overlay visualization",
            "Best and worst case review",
          ]).map((step, index) => (
            <PipelineStep
              key={step}
              index={index}
              label={step}
            />
          ))}
        </div>
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        {status?.disclaimer ??
          "This MRI segmentation module is for research and educational use only and is not intended for clinical diagnosis."}
      </section>
    </AppShell>
  );
}

function ImagePanel({
  title,
  src,
  compact = false,
  large = false,
}: {
  title: string;
  src: string;
  compact?: boolean;
  large?: boolean;
}) {
  const heightClass = large
    ? "h-[560px]"
    : compact
      ? "h-80"
      : "h-96";

  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div
        className={[
          "flex items-center justify-center bg-slate-950 p-4",
          heightClass,
        ].join(" ")}
      >
        <img
          src={src}
          alt={title}
          className="max-h-full max-w-full object-contain"
        />
      </div>

      <figcaption className="border-t border-slate-100 bg-white px-4 py-3 text-center text-sm font-semibold text-slate-700">
        {title}
      </figcaption>
    </figure>
  );
}

function InfoPanel({
  title,
  icon: Icon,
  values,
}: {
  title: string;
  icon: typeof BrainCircuit;
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

function PipelineStep({
  index,
  label,
}: {
  index: number;
  label: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold text-blue-600">
        {String(index + 1).padStart(2, "0")}
      </p>

      <p className="mt-2 text-sm font-semibold text-slate-800">
        {label}
      </p>
    </div>
  );
}

function formatMetric(value?: number): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatMetricOrNA(value?: number | null): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatSafeCount(value?: number | string | unknown[] | null): string {
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