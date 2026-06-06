type StatusBadgeProps = {
  active?: boolean;
  label?: string;
};

export default function StatusBadge({
  active = true,
  label,
}: StatusBadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1",
        "text-xs font-semibold",
        active
          ? "bg-emerald-50 text-emerald-700"
          : "bg-rose-50 text-rose-700",
      ].join(" ")}
    >
      <span
        className={[
          "h-1.5 w-1.5 rounded-full",
          active ? "bg-emerald-500" : "bg-rose-500",
        ].join(" ")}
      />
      {label ?? (active ? "Active" : "Unavailable")}
    </span>
  );
}