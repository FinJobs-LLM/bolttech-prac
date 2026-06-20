import { useState } from "react";

// Generic sortable table. `columns` is [{key, label, fmt?, deEmph?}].
export default function DataTable({ columns, rows, isBest }) {
  const [sortKey, setSortKey] = useState(null);
  const [asc, setAsc] = useState(false);

  const sorted = [...rows];
  if (sortKey) {
    sorted.sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === "number" && typeof y === "number") return asc ? x - y : y - x;
      return asc ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
    });
  }

  const toggle = (key) => {
    if (sortKey === key) setAsc(!asc);
    else { setSortKey(key); setAsc(false); }
  };

  return (
    <table>
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c.key} onClick={() => toggle(c.key)} className={c.deEmph ? "de-emph" : ""}>
              {c.label}{sortKey === c.key ? (asc ? " ▲" : " ▼") : ""}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((r, i) => (
          <tr key={i} className={isBest && isBest(r) ? "best" : ""}>
            {columns.map((c) => (
              <td key={c.key} className={c.deEmph ? "de-emph" : ""}>
                {c.fmt ? c.fmt(r[c.key], r) : r[c.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
