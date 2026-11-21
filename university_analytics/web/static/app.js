"use strict";
const form = document.querySelector("#data-form");
const statusNode = document.querySelector("#status");
const summary = document.querySelector("#summary");
const heading = document.querySelector("#result-heading");
const download = document.querySelector("#download");
let outputCsv = "";
function table(container, headers, rows) {
  const result = document.createElement("table");
  const head = result.createTHead().insertRow();
  headers.forEach(label => { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = label; head.append(cell); });
  const body = result.createTBody();
  rows.forEach(row => { const tr = body.insertRow(); row.forEach(value => { tr.insertCell().textContent = value ?? "—"; }); });
  container.replaceChildren(result);
}
function note(text) { const node = document.createElement("p"); node.textContent = text; summary.append(node); }
async function run(action) {
  if (!form.reportValidity()) return;
  const buttons = form.querySelectorAll("button");
  buttons.forEach(button => { button.disabled = true; });
  statusNode.textContent = "Обработка файла…";
  statusNode.classList.remove("error"); summary.replaceChildren(); document.querySelector("#table").replaceChildren();
  outputCsv = ""; download.hidden = true;
  try {
    const response = await fetch(`/api/${action}`, {method: "POST", body: new FormData(form), headers: {"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не удалось обработать файл.");
    if (action === "inspect") {
      heading.textContent = "Профиль данных";
      statusNode.textContent = `Строк: ${data.row_count}. Колонок: ${data.column_count}. Полных повторов: ${data.duplicate_rows}.`;
      note(`Происхождение: ${data.data_origin === "synthetic" ? "симуляционные данные" : "по файлу не подтверждено"}.`);
      table(document.querySelector("#table"), ["Колонка", "Тип", "Пропуски", "NaN / ∞", "Разных значений"], Object.entries(data.column_profiles).map(([name, item]) => [name, item.kind === "numeric" ? "Число" : "Категория", item.missing, item.nonfinite, item.unique]));
    } else {
      heading.textContent = "Прогнозы готовы";
      statusNode.textContent = `Обработано строк: ${data.report.rows}. Ниже первые ${data.preview.length}; полный результат доступен в CSV.`;
      note(data.report.intended_use === "development_demo_only" ? "Демонстрационная модель на симуляции. Результаты не являются реальной оценкой сотрудников." : "Пригодность модели для реальных решений требует отдельной проверки.");
      if (data.preview.length) {
        const keys = Object.keys(data.preview[0]);
        const names = {source_row: "Строка CSV", predicted_class: "Прогноз"};
        const labels = keys.map(key => names[key] || (data.report.probability_columns[key] ? "Вероятность: " + data.report.probability_columns[key] : key));
        table(document.querySelector("#table"), labels, data.preview.map(row => keys.map(key => row[key])));
      }
      outputCsv = data.csv; download.hidden = false;
    }
  } catch (error) { statusNode.textContent = error.message; statusNode.classList.add("error"); }
  finally { buttons.forEach(button => { button.disabled = false; }); }
}
form.addEventListener("submit", event => { event.preventDefault(); run("predict"); });
document.querySelector("#inspect").addEventListener("click", () => run("inspect"));
download.addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob(["\ufeff", outputCsv], {type: "text/csv;charset=utf-8"}));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = "predictions.csv"; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
(async () => {
  try {
    const response = await fetch("/api/experiments");
    if (!response.ok) throw new Error("Не удалось загрузить эксперименты.");
    const data = await response.json();
    const select = document.querySelector("#experiment"); select.replaceChildren();
    const rows = [], names = {dummy: "Константная", linear: "Линейная", neural: "Нейросеть"};
    data.experiments.forEach(item => {
      if (item.has_portable) { const option = document.createElement("option"); option.value = item.id; option.textContent = item.id + (item.data_origin === "synthetic" ? " · симуляция" : ""); select.append(option); }
      Object.entries(item.metrics).forEach(([name, score]) => rows.push([names[name] || name, item.data_origin, score.validation.macro_f1?.toFixed(4), score.test.macro_f1?.toFixed(4)]));
    });
    if (!select.options.length) { const option = document.createElement("option"); option.value = ""; option.textContent = "Загрузите собственную JSON-модель"; select.append(option); }
    table(document.querySelector("#experiments"), ["Модель", "Данные", "Validation macro-F1", "Test macro-F1"], rows);
  } catch (error) { document.querySelector("#experiments").textContent = error.message; }
})();
