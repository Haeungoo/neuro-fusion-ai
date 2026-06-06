import type { LucideIcon } from "lucide-react";

type StatCardProps = {
  icon: LucideIcon;
  label: string;
  value: string;
  description: string;
  iconClassName?: string;
};

export default function StatCard({
  icon: Icon,
  label,
  value,
  description,
  iconClassName = "bg-teal-50 text-teal-600",
}: StatCardProps) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-4">
        <div
          className={`flex h-12 w-12 items-center justify-center rounded-xl ${iconClassName}`}
        >
          <Icon size={23} strokeWidth={1.8} />
        </div>

        <div>
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <p className="mt-0.5 text-2xl font-bold text-slate-900">{value}</p>
          <p className="mt-0.5 text-xs text-slate-500">{description}</p>
        </div>
      </div>
    </article>
  );
}