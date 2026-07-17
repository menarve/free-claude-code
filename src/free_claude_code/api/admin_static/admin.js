const state = {
  config: null,
  fields: new Map(),
  localStatus: new Map(),
  modelOptions: [],
  activeView: "chat",
  usageSort: "list",
};

const MASKED_SECRET = "********";
// A model role may point at the derivation chain instead of a fixed model.
// The env stores the sentinel ref; the UI shows a friendly name.
const DERIVATION_MODEL_REF = "menarve/derivation";
const DERIVATION_DISPLAY_NAME = "Derivación Menarve";
const VIEW_GROUPS = [
  {
    id: "chat",
    label: "Chat",
    title: "Chat",
    sections: [],
    containerId: "chatSections",
  },
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    sections: ["providers", "runtime"],
    containerId: "providersSections",
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    sections: ["models", "thinking", "web_tools"],
    containerId: "modelConfigSections",
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
  },
  {
    id: "usage",
    label: "Usage",
    title: "Usage",
    sections: [],
    containerId: "usageSections",
  },
];

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: "default",
    template: "template",
    repo_env: "repo .env",
    managed_env: "",
    explicit_env_file: "FCC_ENV_FILE",
    process: "process env",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const label = sourceLabel(field.source);
  if (label) {
    parts.push(label);
  }
  if (field.locked) {
    parts.push("locked");
  }
  return parts.join(" ");
}

function statusClass(status) {
  if (["configured", "reachable", "running"].includes(status)) return "ok";
  if (["missing_key", "missing_url", "unknown"].includes(status)) return "warn";
  if (["offline", "error"].includes(status)) return "error";
  return "neutral";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function load() {
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  state.config = config;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  renderNav();
  renderProviders(config.provider_status);
  renderSections(config.sections, config.fields);
  syncModelDatalist();
  renderUsage(await api("/admin/api/usage"));
  byId("configPath").textContent = config.paths.managed;
  await validate(false);
  await refreshLocalStatus();
  updateDirtyState();
  showMessage("");
}

function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `nav-link${index === 0 ? " active" : ""}`;
    button.dataset.view = view.id;
    button.textContent = view.label;
    if (index === 0) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      setActiveView(view.id, { scroll: true });
    });
    nav.appendChild(button);
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView =
    VIEW_GROUPS.find((view) => view.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;
  byId("pageTitle").textContent = activeView.title;

  // The config action bar (Validate/Apply) is irrelevant while chatting.
  document.body.classList.toggle("chat-active", activeView.id === "chat");

  document.querySelectorAll(".nav-link").forEach((link) => {
    const selected = link.dataset.view === activeView.id;
    link.classList.toggle("active", selected);
    if (selected) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".admin-view").forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle("active", selected);
    view.hidden = !selected;
  });

  if (scroll) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function renderProviders(providerStatus) {
  const grid = byId("providerGrid");
  grid.innerHTML = "";
  providerStatus.forEach((provider) => {
    const card = document.createElement("article");
    card.className = "provider-card";
    card.dataset.provider = provider.provider_id;

    const title = document.createElement("div");
    title.className = "provider-title";
    title.innerHTML = `<strong>${provider.display_name || provider.provider_id}</strong>`;

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;
    title.appendChild(pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    meta.textContent =
      provider.kind === "local"
        ? provider.base_url || "No local URL configured"
        : provider.credential_env;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "test-button";
    button.textContent = provider.kind === "local" ? "Test" : "Refresh models";
    button.addEventListener("click", () => testProvider(provider.provider_id, button));

    card.append(title, meta, button);
    grid.appendChild(card);
  });
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector(".status-pill");
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  if (metaText) {
    card.querySelector(".provider-meta").textContent = metaText;
  }
}

function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    if (container) container.innerHTML = "";
  });

  const sectionById = new Map(sections.map((section) => [section.id, section]));
  const bySection = new Map();
  sections.forEach((section) => bySection.set(section.id, []));
  fields.forEach((field) => {
    if (!bySection.has(field.section)) bySection.set(field.section, []);
    bySection.get(field.section).push(field);
  });

  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    view.sections.forEach((sectionId) => {
      const section = sectionById.get(sectionId);
      const sectionFields = bySection.get(sectionId) || [];
      if (!section || sectionFields.length === 0) return;

      const sectionEl = document.createElement("section");
      sectionEl.className = "settings-section";
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement("div");
      heading.className = "section-heading";
      heading.innerHTML = `<div><h3>${section.label}</h3><p>${section.description}</p></div>`;
      sectionEl.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "field-grid";
      sectionFields.forEach((field) => {
        grid.appendChild(renderField(field));
      });
      sectionEl.appendChild(grid);

      if (sectionFields.some((field) => field.advanced)) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "ghost-button advanced-toggle";
        toggle.textContent = "Show advanced";
        toggle.addEventListener("click", () => {
          const showing = sectionEl.classList.toggle("show-advanced");
          toggle.textContent = showing ? "Hide advanced" : "Show advanced";
        });
        sectionEl.appendChild(toggle);
      }

      container.appendChild(sectionEl);
    });
  });
}

function renderUsage(usage) {
  const container = byId("usageSections");
  container.innerHTML = "";

  const section = document.createElement("section");
  section.className = "settings-section";

  const heading = document.createElement("div");
  heading.className = "section-heading";
  heading.innerHTML =
    "<div><h3>Model usage</h3><p>Requests, input tokens, and errors for every model the derivation system can use with your configured API keys (paid OpenRouter models excluded).</p></div>";
  section.appendChild(heading);

  const controls = document.createElement("div");
  controls.className = "usage-controls";

  const refreshButton = document.createElement("button");
  refreshButton.type = "button";
  refreshButton.className = "ghost-button";
  refreshButton.textContent = "Refresh";
  refreshButton.addEventListener("click", async () => {
    renderUsage(await api("/admin/api/usage"));
  });
  controls.appendChild(refreshButton);

  const sortLabel = document.createElement("span");
  sortLabel.className = "usage-sort-label";
  sortLabel.textContent = "Sort by:";
  controls.appendChild(sortLabel);

  [
    ["list", "Derivation order"],
    ["usage", "Most used"],
  ].forEach(([value, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `ghost-button usage-sort-button${
      state.usageSort === value ? " active" : ""
    }`;
    button.textContent = label;
    button.addEventListener("click", () => {
      state.usageSort = value;
      renderUsage(usage);
    });
    controls.appendChild(button);
  });

  section.appendChild(controls);

  const eligible = usage.eligible || [];
  const derivationOrder = new Map(eligible.map((ref, index) => [ref, index]));
  const emptyStats = { requests: 0, errors: 0, input_tokens: 0, last_used_at: null };
  const rows = eligible.map((modelRef) => ({
    modelRef,
    stats: (usage.models && usage.models[modelRef]) || emptyStats,
  }));

  if (state.usageSort === "usage") {
    rows.sort((a, b) => b.stats.requests - a.stats.requests);
  } else {
    rows.sort(
      (a, b) => derivationOrder.get(a.modelRef) - derivationOrder.get(b.modelRef),
    );
  }

  const countLabel = document.createElement("p");
  countLabel.className = "field-description";
  countLabel.textContent = `${rows.length} models available for derivation.`;
  section.appendChild(countLabel);

  if (rows.length === 0) {
    const empty = document.createElement("p");
    empty.className = "field-description";
    empty.textContent = "No models available yet.";
    section.appendChild(empty);
  } else {
    const table = document.createElement("table");
    table.className = "usage-table";
    const thead = document.createElement("thead");
    thead.innerHTML =
      "<tr><th>Model</th><th>Requests</th><th>Input tokens</th><th>Errors</th><th>Last used</th></tr>";
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach(({ modelRef, stats }) => {
      const row = document.createElement("tr");
      const lastUsed = stats.last_used_at
        ? new Date(stats.last_used_at).toLocaleString()
        : "-";
      row.innerHTML = `
        <td>${modelRef}</td>
        <td>${stats.requests}</td>
        <td>${stats.input_tokens.toLocaleString()}</td>
        <td>${stats.errors}</td>
        <td>${lastUsed}</td>
      `;
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    section.appendChild(table);
  }

  container.appendChild(section);
}

function renderField(field) {
  const wrapper = document.createElement("div");
  wrapper.className = `field${field.advanced ? " advanced-field" : ""}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement("label");
  label.htmlFor = `field-${field.key}`;
  const labelText = document.createElement("span");
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const source = sourceText(field);
  if (source) {
    const sourceEl = document.createElement("span");
    sourceEl.className = "field-source";
    sourceEl.textContent = source;
    label.appendChild(sourceEl);
  }

  const input = inputForField(field);
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = field.value || "";
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.disabled = field.locked;
  input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);

  wrapper.append(label, input);
  if (field.description) {
    const description = document.createElement("div");
    description.className = "field-description";
    description.textContent = field.description;
    wrapper.appendChild(description);
  }
  return wrapper;
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "tri_boolean") {
    const select = document.createElement("select");
    [
      ["", "Inherit"],
      ["true", "Enabled"],
      ["false", "Disabled"],
    ].forEach(([value, label]) => select.appendChild(option(value, label)));
    select.value = field.value || "";
    return select;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((value) => select.appendChild(option(value, value)));
    select.value = field.value || field.options[0] || "";
    return select;
  }

  if (field.type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.value = field.value || "";
    return textarea;
  }

  const input = document.createElement("input");
  input.type = field.type === "number" ? "number" : "text";
  if (field.type === "secret") {
    input.type = "password";
    input.placeholder = field.configured
      ? "Configured - enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else if (field.key.startsWith("MODEL") && field.value === DERIVATION_MODEL_REF) {
    input.value = DERIVATION_DISPLAY_NAME;
  } else {
    input.value = field.value || "";
  }
  if (field.key.startsWith("MODEL")) {
    input.setAttribute("list", "model-options");
  }
  return input;
}

function option(value, label) {
  const optionEl = document.createElement("option");
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
}

function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  if (input.value === DERIVATION_DISPLAY_NAME) return DERIVATION_MODEL_REF;
  return input.value;
}

function changedValues() {
  const values = {};
  document.querySelectorAll("[data-key]").forEach((input) => {
    if (input.disabled || !input.matches("input, select, textarea")) return;
    const value = readFieldValue(input);
    if (value !== input.dataset.original) {
      values[input.dataset.key] = value;
    }
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId("dirtyState").textContent =
    count === 0 ? "No changes" : `${count} unsaved change${count === 1 ? "" : "s"}`;
  byId("applyButton").disabled = count === 0;
}

async function validate(showResult = true) {
  const result = await api("/admin/api/config/validate", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (showResult) {
    showValidationResult(result);
  }
  return result;
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage("Config shape is valid", "ok");
  } else {
    showMessage(result.errors.join("; "), "error");
  }
}

async function apply() {
  const result = await api("/admin/api/config/apply", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (!result.applied) {
    showValidationResult(result);
    return;
  }
  const restart = result.restart || {};
  if (restart.required && restart.automatic) {
    showMessage("Applied. Restarting server...", "ok");
    byId("applyButton").disabled = true;
    setTimeout(() => {
      window.location.href = restart.admin_url || "/admin";
    }, 1600);
    return;
  }
  const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
  await load();
  showMessage(
    pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Applied",
    "ok",
  );
}

async function refreshLocalStatus() {
  const result = await api("/admin/api/providers/local-status");
  result.providers.forEach((provider) => {
    state.localStatus.set(provider.provider_id, provider);
    const meta = provider.status_code
      ? `${provider.base_url} returned HTTP ${provider.status_code}`
      : provider.base_url;
    updateProviderCard(provider.provider_id, provider.status, provider.label, meta);
  });
}

async function testProvider(providerId, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Testing";
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });
    if (result.ok) {
      updateProviderCard(
        providerId,
        "reachable",
        `${result.models.length} models`,
        result.models.slice(0, 3).join(", ") || "No models returned",
      );
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((model) => `${providerId}/${model}`),
        ]),
      ).sort();
      syncModelDatalist();
    } else {
      updateProviderCard(providerId, "offline", result.error_type, result.error_type);
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function syncModelDatalist() {
  let datalist = byId("model-options");
  if (!datalist) {
    datalist = document.createElement("datalist");
    datalist.id = "model-options";
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = "";
  datalist.appendChild(option(DERIVATION_DISPLAY_NAME, DERIVATION_DISPLAY_NAME));
  state.modelOptions.forEach((model) => datalist.appendChild(option(model, model)));
}

function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
}

const chatState = { history: [], streaming: false };

function appendChatMessage(role, text) {
  const container = byId("chatMessages");
  const empty = container.querySelector(".chat-empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = `chat-msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return { row, bubble };
}

async function sendChat() {
  if (chatState.streaming) return;
  const input = byId("chatInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  chatState.history.push({ role: "user", content: text });
  appendChatMessage("user", text);

  chatState.streaming = true;
  const sendButton = byId("chatSend");
  sendButton.disabled = true;
  const { row, bubble } = appendChatMessage("assistant", "");
  bubble.classList.add("chat-typing");
  let assistantText = "";
  let model = null;
  try {
    const response = await fetch("/admin/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatState.history }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const dataLine = part.split("\n").find((line) => line.startsWith("data:"));
        if (!dataLine) continue;
        let data;
        try {
          data = JSON.parse(dataLine.slice(5).trim());
        } catch (error) {
          continue;
        }
        if (data.type === "message_start") {
          model = ((data.message && data.message.model) || "").split("/").pop();
          if (model && !row.querySelector(".chat-model")) {
            const badge = document.createElement("div");
            badge.className = "chat-model";
            badge.textContent = `⚡ ${model}`;
            row.insertBefore(badge, bubble);
          }
        } else if (
          data.type === "content_block_delta" &&
          data.delta &&
          data.delta.type === "text_delta"
        ) {
          assistantText += data.delta.text;
          bubble.classList.remove("chat-typing");
          bubble.textContent = assistantText;
          byId("chatMessages").scrollTop = byId("chatMessages").scrollHeight;
        } else if (data.type === "error") {
          throw new Error((data.error && data.error.message) || "stream error");
        }
      }
    }
    bubble.classList.remove("chat-typing");
    if (!assistantText) bubble.textContent = "(sin respuesta)";
    chatState.history.push({ role: "assistant", content: assistantText });
  } catch (error) {
    bubble.classList.remove("chat-typing");
    bubble.textContent = `Error: ${error.message}`;
    bubble.classList.add("chat-error");
    chatState.history.pop();
  } finally {
    chatState.streaming = false;
    sendButton.disabled = false;
  }
}

function setupChat() {
  const form = byId("chatForm");
  const input = byId("chatInput");
  if (!form || !input) return;
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendChat();
  });
}

byId("validateButton").addEventListener("click", () => validate(true));
byId("applyButton").addEventListener("click", apply);
setupChat();

load().catch((error) => {
  showMessage(error.message, "error");
});
