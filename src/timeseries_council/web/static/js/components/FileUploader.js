/**
 * FileUploader Component - Drag and drop file upload
 */

const FileUploader = {
    name: 'FileUploader',

    props: {
        multiple: {
            type: Boolean,
            default: true
        },
        accept: {
            type: String,
            default: '.csv,.xlsx,.xls,.json'
        }
    },

    emits: ['files-selected', 'file-removed'],

    template: `
        <div class="file-uploader">
            <div
                class="drop-zone"
                :class="{
                    'drag-over': isDragOver,
                    'has-files': files.length > 0
                }"
                @dragover.prevent="onDragOver"
                @dragleave.prevent="onDragLeave"
                @drop.prevent="onDrop"
                @click="openFilePicker"
            >
                <input
                    ref="fileInput"
                    type="file"
                    :accept="accept"
                    :multiple="multiple"
                    @change="onFileSelect"
                    class="hidden-input"
                >

                <div v-if="files.length === 0" class="drop-prompt">
                    <div class="drop-icon">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <p class="drop-text">Drop files here</p>
                    <p class="drop-subtext">or click to browse</p>
                </div>

                <div v-else class="file-list">
                    <div
                        v-for="(file, index) in files"
                        :key="file.name + index"
                        class="file-item"
                        @click.stop
                    >
                        <span class="file-type-icon">{{ getFileIcon(file.name) }}</span>
                        <div class="file-details">
                            <span class="file-name">{{ file.name }}</span>
                            <span class="file-size">{{ formatSize(file.size) }}</span>
                        </div>
                        <button
                            class="remove-btn"
                            @click.stop="removeFile(index)"
                            title="Remove file"
                        >
                            ×
                        </button>
                    </div>
                </div>
            </div>

            <div v-if="files.length > 0" class="file-actions">
                <button class="clear-btn" @click="clearAll">
                    Clear all
                </button>
            </div>
        </div>
    `,

    data() {
        return {
            files: [],
            isDragOver: false
        };
    },

    methods: {
        openFilePicker() {
            this.$refs.fileInput.click();
        },

        onDragOver(e) {
            this.isDragOver = true;
        },

        onDragLeave(e) {
            this.isDragOver = false;
        },

        onDrop(e) {
            this.isDragOver = false;
            const droppedFiles = Array.from(e.dataTransfer.files);
            this.addFiles(droppedFiles);
        },

        onFileSelect(e) {
            const selectedFiles = Array.from(e.target.files);
            this.addFiles(selectedFiles);
            // Reset input for re-selection
            e.target.value = '';
        },

        addFiles(newFiles) {
            const validExtensions = this.accept.split(',').map(e => e.trim());
            const validFiles = newFiles.filter(file =>
                validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
            );

            if (this.multiple) {
                this.files = [...this.files, ...validFiles];
            } else {
                this.files = validFiles.slice(0, 1);
            }

            this.$emit('files-selected', this.files);
        },

        removeFile(index) {
            const removed = this.files.splice(index, 1)[0];
            this.$emit('file-removed', removed);
            this.$emit('files-selected', this.files);
        },

        clearAll() {
            this.files = [];
            this.$emit('files-selected', []);
        },

        formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / 1048576).toFixed(1) + ' MB';
        },

        getFileIcon(filename) {
            const ext = filename.split('.').pop().toLowerCase();
            const icons = {
                csv: '📊',
                xlsx: '📗',
                xls: '📗',
                json: '📋'
            };
            return icons[ext] || '📄';
        }
    }
};

if (typeof window !== 'undefined') {
    window.FileUploader = FileUploader;
}
