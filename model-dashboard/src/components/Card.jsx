export default function Card({ label, value, small, hint }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className={"value" + (small ? " small" : "")}>{value}</div>
      {hint && <div className="note" style={{ marginTop: 4 }}>{hint}</div>}
    </div>
  );
}
