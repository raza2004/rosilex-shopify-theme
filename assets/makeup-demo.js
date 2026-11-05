/**
 * AI product try on (sections/makeup-demo.liquid).
 * Sends the photo and the chosen product's details to the makeup service, then shows a before and after
 * slider with the identity score. Messages come from data-msg-* attributes (locale files).
 */
(() => {
  if (customElements.get('makeup-demo')) return;

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const REQUEST_TIMEOUT_MS = 90000;

  // API error codes (makeup-service/app/main.py) to message attributes.
  const ERROR_MESSAGES = {
    file_too_large: 'msgFileSize',
    unsupported_type: 'msgFileType',
    unreadable_image: 'msgFileType',
    empty_file: 'msgFileType',
    no_face: 'msgNoFace',
    multiple_faces: 'msgMultipleFaces',
    rate_limited: 'msgRateLimited',
    daily_limit: 'msgDailyLimit',
    generation_busy: 'msgBusy',
    not_configured: 'msgBusy',
  };

  // Seconds since the visit started (set in layout/theme.liquid), or null if unknown.
  function secondsOnSite() {
    try {
      const start = Number(sessionStorage.getItem('visit-start'));
      return start ? Math.max(0, Math.round((Date.now() - start) / 1000)) : null;
    } catch {
      return null;
    }
  }

  class MakeupDemo extends HTMLElement {
    connectedCallback() {
      if (this.ready) return;
      this.ready = true;

      this.form = this.querySelector('[data-form]');
      this.fileInput = this.querySelector('[data-file]');
      this.drop = this.querySelector('[data-drop]');
      this.preview = this.querySelector('[data-preview]');
      this.compare = this.querySelector('[data-compare]');
      this.before = this.querySelector('[data-before]');
      this.after = this.querySelector('[data-after]');
      this.slider = this.querySelector('[data-slider]');
      this.loading = this.querySelector('[data-loading]');
      this.submit = this.querySelector('[data-submit]');
      this.changeButton = this.querySelector('[data-change]');
      this.errorText = this.querySelector('[data-error]');
      this.result = this.querySelector('[data-result]');
      this.shadeField = this.querySelector('[data-shade-field]');
      this.shade = this.querySelector('[data-shade]');
      try {
        this.products = JSON.parse(this.querySelector('[data-products]')?.textContent || '{}');
      } catch {
        this.products = {};
      }

      this.fileInput.addEventListener('change', () => this.setFile(this.fileInput.files[0]));
      this.form.addEventListener('submit', (event) => {
        event.preventDefault();
        this.apply();
      });
      this.changeButton.addEventListener('click', () => this.fileInput.click());
      this.slider.addEventListener('input', () => this.setSplit(this.slider.value));
      this.form.addEventListener('change', (event) => {
        if (event.target.name === 'product') this.showShades(event.target.value);
      });

      this.drop.addEventListener('dragover', (event) => {
        event.preventDefault();
        this.drop.classList.add('is-dragging');
      });
      this.drop.addEventListener('dragleave', () => this.drop.classList.remove('is-dragging'));
      this.drop.addEventListener('drop', (event) => {
        event.preventDefault();
        this.drop.classList.remove('is-dragging');
        this.setFile(event.dataTransfer.files[0]);
      });

      this.setSplit(this.slider.value);
    }

    disconnectedCallback() {
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    }

    message(key, values = {}) {
      return Object.entries(values).reduce(
        (text, [name, value]) => text.replace(`#${name}`, value),
        this.dataset[key] || this.dataset.msgGeneric
      );
    }

    setFile(file) {
      this.showError(null);
      if (!file) return;
      if (!ALLOWED_TYPES.includes(file.type)) return this.showError(this.message('msgFileType'));
      if (file.size > Number(this.dataset.maxBytes)) return this.showError(this.message('msgFileSize'));

      this.file = file;
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = URL.createObjectURL(file);
      this.preview.src = this.previewUrl;
      this.preview.hidden = false;
      this.drop.classList.add('has-photo');
      this.drop.hidden = false;
      this.compare.hidden = true;
      this.result.hidden = true;
      this.changeButton.hidden = false;
    }

    showShades(handle) {
      const shades = this.products[handle]?.shades || [];
      this.shade.replaceChildren(...shades.map((title) => new Option(title, title)));
      this.shadeField.hidden = shades.length === 0;
    }

    selectedProduct() {
      const handle = new FormData(this.form).get('product');
      const product = handle && this.products[handle];
      if (!product) return null;
      const { shades, ...details } = product;
      return { handle, ...details, shade: shades.length ? this.shade.value : '' };
    }

    setSplit(value) {
      this.compare.style.setProperty('--split', `${value}%`);
    }

    async apply() {
      this.showError(null);
      if (!this.file) return this.showError(this.message('msgChoosePhoto'));
      const product = this.selectedProduct();
      if (!product) return this.showError(this.message('msgChooseProduct'));
      if (!this.querySelector('[data-consent]').checked) return this.showError(this.message('msgConsent'));

      const { handle, ...details } = product;
      this.handle = handle;
      const body = new FormData();
      body.append('image', this.file);
      body.append('product', JSON.stringify(details));
      const seconds = secondsOnSite();
      if (seconds !== null) body.append('seconds_on_site', String(seconds));

      this.setBusy(true);
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      try {
        const response = await fetch(`${this.dataset.api}/api/makeup`, { method: 'POST', body, signal: controller.signal });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          const key = ERROR_MESSAGES[data.error?.code] || (response.status === 429 ? 'msgRateLimited' : 'msgGeneric');
          return this.showError(this.message(key));
        }
        this.showResult(data);
      } catch {
        this.showError(this.message('msgNetwork'));
      } finally {
        clearTimeout(timer);
        this.setBusy(false);
      }
    }

    setBusy(busy) {
      this.loading.hidden = !busy;
      this.submit.disabled = busy;
      this.setAttribute('aria-busy', String(busy));
    }

    showResult(data) {
      const resultUrl = `data:${data.mime_type};base64,${data.image}`;
      this.before.src = this.previewUrl;
      this.after.src = resultUrl;
      this.slider.value = 50;
      this.setSplit(50);
      this.drop.hidden = true;
      this.compare.hidden = false;

      const score = typeof data.identity_score === 'number' ? data.identity_score.toFixed(2) : null;
      const threshold = Number(data.threshold).toFixed(2);
      const scoreText = this.querySelector('[data-score]');
      scoreText.textContent = score === null
        ? this.message('msgNoFaceResult')
        : this.message(data.passed ? 'msgPassed' : 'msgFailed', { score, threshold });
      scoreText.classList.toggle('is-failed', !data.passed);

      this.querySelector('[data-attempts]').textContent = this.message('msgAttempts', { count: data.attempts, max: data.max_attempts });
      const triesLeft = this.querySelector('[data-tries-left]');
      triesLeft.hidden = typeof data.tries_left !== 'number';
      triesLeft.textContent = triesLeft.hidden ? '' : this.message('msgTriesLeft', { count: data.tries_left });
      this.querySelector('[data-threshold]').textContent = this.message('msgThreshold', { threshold });
      this.querySelector('[data-engine]').textContent = this.message(data.engine === 'gemini' ? 'msgEngineGemini' : 'msgEngineOpenai');
      const download = this.querySelector('[data-download]');
      download.href = resultUrl;
      download.download = `rosilex-${this.handle}.${data.mime_type === 'image/jpeg' ? 'jpg' : 'png'}`;
      this.result.hidden = false;
    }

    showError(text) {
      this.errorText.hidden = !text;
      this.errorText.textContent = text || '';
    }
  }

  customElements.define('makeup-demo', MakeupDemo);
})();
