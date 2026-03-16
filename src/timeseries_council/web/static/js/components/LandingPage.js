/**
 * LandingPage Component - File upload and mode selection
 */

const LandingPage = {
    name: 'LandingPage',

    template: `
        <div class="landing-page">
            <header class="landing-header">
                <h1>Time Series Council</h1>
                <p class="subtitle">AI-Powered Analysis with Multi-Model Intelligence</p>
            </header>

            <!-- File Upload Zone -->
            <div
                class="upload-zone"
                :class="{ 'drag-over': isDragOver, 'has-files': files.length > 0 }"
                @dragover.prevent="isDragOver = true"
                @dragleave.prevent="isDragOver = false"
                @drop.prevent="handleDrop"
                @click="triggerFileInput"
            >
                <input
                    ref="fileInput"
                    type="file"
                    accept=".csv,.xlsx,.xls,.json"
                    multiple
                    @change="handleFileSelect"
                    style="display: none"
                >

                <div v-if="files.length === 0" class="upload-prompt">
                    <div class="upload-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <p class="upload-text">Drop your CSV file here or click to browse</p>
                    <p class="upload-hint">Supports CSV, Excel, and JSON formats</p>
                </div>

                <div v-else class="files-list">
                    <div
                        v-for="(file, index) in files"
                        :key="index"
                        class="file-item"
                        @click.stop
                    >
                        <div class="file-icon">📊</div>
                        <div class="file-info">
                            <span class="file-name">{{ file.name }}</span>
                            <span class="file-size">{{ formatFileSize(file.size) }}</span>
                        </div>
                        <button class="remove-file" @click.stop="removeFile(index)">×</button>
                    </div>
                </div>
            </div>

            <!-- Data Preview (when file loaded) -->
            <div v-if="dataPreview" class="data-preview">
                <h3>Data Preview</h3>
                <div class="preview-info">
                    <span>{{ dataPreview.rows }} rows</span>
                    <span>{{ dataPreview.columns.length }} columns</span>
                </div>
                <div class="preview-columns">
                    <span
                        v-for="col in dataPreview.columns"
                        :key="col"
                        class="column-tag"
                        :class="{ 'selected': col === targetColumn }"
                        @click="targetColumn = col"
                    >
                        {{ col }}
                    </span>
                </div>
                <p class="column-hint">Click a column to set as analysis target</p>
            </div>

            <!-- Configuration -->
            <div class="config-section">
                <h3>Configuration</h3>

                <div class="config-row">
                    <div class="config-item">
                        <label>Target Column</label>
                        <input
                            v-model="targetColumn"
                            type="text"
                            placeholder="e.g., sales, temperature"
                        >
                    </div>

                    <div class="config-item">
                        <label>LLM Provider</label>
                        <select v-model="provider">
                            <option value="anthropic">Claude (Anthropic)</option>
                            <option value="gemini">Gemini (Google)</option>
                            <option value="openai">GPT-4 (OpenAI)</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="qwen">Qwen</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Mode Selection -->
            <div class="mode-section">
                <h3>Analysis Mode</h3>

                <div class="mode-cards">
                    <div
                        class="mode-card"
                        :class="{ 'selected': mode === 'standard' }"
                        @click="mode = 'standard'"
                    >
                        <div class="mode-icon">🧠</div>
                        <h4>Smart Analysis</h4>
                        <p>Multi-model AI analysis. Automatically selects 3-5 best models for your data.</p>
                        <span class="mode-badge">Recommended</span>
                    </div>

                    <div
                        class="mode-card"
                        :class="{ 'selected': mode === 'council' }"
                        @click="mode = 'council'"
                    >
                        <div class="mode-icon">👥</div>
                        <h4>Council</h4>
                        <p>Multi-perspective deliberation. Get insights from analyst, risk, and business viewpoints.</p>
                    </div>
                </div>
            </div>

            <!-- Context Input (Optional) -->
            <div class="context-section">
                <label>Additional Context <span class="optional">(Optional)</span></label>
                <textarea
                    v-model="userContext"
                    placeholder="Add any context about your data, such as: 'This is daily sales data for a retail store. Holidays may affect the patterns.'"
                    rows="3"
                ></textarea>
            </div>

            <!-- Error Display -->
            <div v-if="error" class="error-message">
                {{ error }}
            </div>

            <!-- Start Button -->
            <div class="start-section">
                <button
                    class="btn-start"
                    :disabled="!canStart || isLoading"
                    @click="startSession"
                >
                    <span v-if="isLoading" class="spinner"></span>
                    {{ isLoading ? 'Starting...' : 'Start Analysis' }}
                </button>
            </div>
        </div>
    `,

    data() {
        return {
            files: [],
            isDragOver: false,
            dataPreview: null,
            targetColumn: 'sales',
            provider: 'anthropic',
            mode: 'standard',
            userContext: '',
            error: null,
            isLoading: false
        };
    },

    computed: {
        canStart() {
            return this.files.length > 0 && this.targetColumn;
        }
    },

    methods: {
        triggerFileInput() {
            this.$refs.fileInput.click();
        },

        handleFileSelect(event) {
            const newFiles = Array.from(event.target.files);
            this.addFiles(newFiles);
        },

        handleDrop(event) {
            this.isDragOver = false;
            const newFiles = Array.from(event.dataTransfer.files);
            this.addFiles(newFiles);
        },

        addFiles(newFiles) {
            const validFiles = newFiles.filter(f =>
                f.name.endsWith('.csv') ||
                f.name.endsWith('.xlsx') ||
                f.name.endsWith('.xls') ||
                f.name.endsWith('.json')
            );

            if (validFiles.length !== newFiles.length) {
                this.error = 'Some files were skipped. Only CSV, Excel, and JSON are supported.';
                setTimeout(() => this.error = null, 3000);
            }

            this.files = [...this.files, ...validFiles];

            // Preview first file
            if (validFiles.length > 0 && !this.dataPreview) {
                this.previewFile(validFiles[0]);
            }
        },

        removeFile(index) {
            this.files.splice(index, 1);
            if (this.files.length === 0) {
                this.dataPreview = null;
            }
        },

        formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        },

        async previewFile(file) {
            try {
                const text = await file.text();
                const lines = text.split('\n').filter(l => l.trim());
                const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

                this.dataPreview = {
                    rows: lines.length - 1,
                    columns: headers
                };

                // Auto-select first numeric-looking column as target
                const numericCols = headers.filter(h =>
                    !h.toLowerCase().includes('date') &&
                    !h.toLowerCase().includes('time') &&
                    !h.toLowerCase().includes('id')
                );
                if (numericCols.length > 0) {
                    this.targetColumn = numericCols[0];
                }
            } catch (e) {
                console.error('Preview error:', e);
            }
        },

        async startSession() {
            if (!this.canStart) return;

            this.isLoading = true;
            this.error = null;

            try {
                // Upload file(s)
                const formData = new FormData();
                this.files.forEach((file, i) => {
                    formData.append('files', file);
                });

                const uploadResponse = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    const err = await uploadResponse.json();
                    throw new Error(err.detail || 'Upload failed');
                }

                const uploadData = await uploadResponse.json();

                // Create session with uploaded file
                const sessionResponse = await fetch('/api/session/upload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_id: uploadData.file_id,
                        target_col: this.targetColumn,
                        provider: this.provider,
                        mode: this.mode,
                        user_context: this.userContext || null
                    })
                });

                if (!sessionResponse.ok) {
                    const err = await sessionResponse.json();
                    throw new Error(err.detail || 'Failed to start session');
                }

                const sessionData = await sessionResponse.json();

                // Emit session started event
                this.$emit('session-started', {
                    ...sessionData,
                    mode: this.mode,
                    userContext: this.userContext
                });

            } catch (e) {
                this.error = e.message;
            } finally {
                this.isLoading = false;
            }
        }
    }
};

// Export for use in app
if (typeof window !== 'undefined') {
    window.LandingPage = LandingPage;
}
