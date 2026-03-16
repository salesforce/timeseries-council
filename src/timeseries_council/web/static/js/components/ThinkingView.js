/**
 * ThinkingView Component - Claude Code style progress view
 * Shows planning, thinking, and execution steps to keep users engaged
 */

const ThinkingView = {
    name: 'ThinkingView',

    props: {
        thinking: {
            type: Object,
            default: () => ({})
        },
        skillResult: {
            type: Object,
            default: null
        },
        isLoading: {
            type: Boolean,
            default: false
        }
    },

    template: `
        <div class="thinking-view" :class="{ 'loading': isLoading, 'expanded': isExpanded }">
            <!-- Main Progress Panel -->
            <div class="progress-panel">
                <!-- Header -->
                <div class="progress-header" @click="isExpanded = !isExpanded">
                    <span class="status-dot" :class="statusClass"></span>
                    <span class="progress-title">{{ headerTitle }}</span>
                    <span class="expand-icon">{{ isExpanded ? '−' : '+' }}</span>
                </div>

                <!-- Expanded Content -->
                <div v-show="isExpanded" class="progress-content">
                    <!-- Task List -->
                    <div class="task-list">
                        <div
                            v-for="(task, index) in tasks"
                            :key="index"
                            class="task-item"
                            :class="task.type"
                        >
                            <!-- Task with checkbox -->
                            <template v-if="task.type === 'task'">
                                <span class="task-checkbox" :class="task.status">
                                    <span v-if="task.status === 'running'" class="spinner-dot"></span>
                                    <span v-else-if="task.status === 'complete'">✓</span>
                                    <span v-else class="checkbox-empty"></span>
                                </span>
                                <span class="task-text">{{ task.text }}</span>
                            </template>

                            <!-- Thinking section (expandable) -->
                            <template v-else-if="task.type === 'thinking'">
                                <div class="thinking-row" @click.stop="task.expanded = !task.expanded">
                                    <span class="status-dot small" :class="task.status === 'complete' ? 'complete' : 'active'"></span>
                                    <span class="thinking-label">Thinking</span>
                                    <span class="thinking-caret">{{ task.expanded ? '∨' : '›' }}</span>
                                </div>
                                <div v-if="task.expanded" class="thinking-detail">
                                    {{ task.text }}
                                </div>
                            </template>

                            <!-- Read file action -->
                            <template v-else-if="task.type === 'read'">
                                <span class="status-dot small" :class="task.status === 'complete' ? 'complete' : 'active'"></span>
                                <span class="action-label">Read</span>
                                <span class="file-name">{{ task.text }}</span>
                            </template>

                            <!-- Execute action -->
                            <template v-else-if="task.type === 'execute'">
                                <span class="status-dot small" :class="task.status === 'complete' ? 'complete' : 'active'"></span>
                                <span class="action-label">Execute</span>
                                <span class="model-name">{{ task.text }}</span>
                            </template>

                            <!-- Status message -->
                            <template v-else-if="task.type === 'status'">
                                <span class="status-indicator">
                                    <span v-if="task.status === 'running'" class="typing-dots">
                                        <span></span><span></span><span></span>
                                    </span>
                                    <span v-else class="status-dot small complete"></span>
                                </span>
                                <span class="status-text">{{ task.text }}</span>
                            </template>
                        </div>
                    </div>

                    <!-- Models Used Summary -->
                    <div v-if="skillResult && skillResult.models_used && skillResult.models_used.length > 0" class="models-summary">
                        <div class="summary-header">
                            <span class="status-dot small complete"></span>
                            <span>Models Used</span>
                        </div>
                        <div class="model-tags">
                            <span v-for="model in skillResult.models_used" :key="model" class="model-tag">
                                {{ model }}
                            </span>
                        </div>
                    </div>

                    <!-- Execution Time -->
                    <div v-if="skillResult && skillResult.execution_time" class="execution-time">
                        Completed in {{ (skillResult.execution_time * 1000).toFixed(0) }}ms
                    </div>
                </div>
            </div>
        </div>
    `,

    data() {
        return {
            isExpanded: true,
            tasks: [],
            thinkingExpanded: false
        };
    },

    computed: {
        statusClass() {
            if (this.isLoading) return 'active';
            if (this.skillResult?.success) return 'complete';
            if (this.skillResult?.success === false) return 'error';
            return 'pending';
        },
        headerTitle() {
            if (this.isLoading) {
                return 'Processing request...';
            }
            if (this.skillResult?.skill_name) {
                return `Used ${this.skillResult.skill_name}`;
            }
            return 'View details';
        }
    },

    watch: {
        isLoading: {
            handler(val) {
                if (val) {
                    this.isExpanded = true;
                    this.startLoadingSequence();
                }
            },
            immediate: true
        },

        skillResult: {
            handler(val) {
                if (val) {
                    this.updateWithResult(val);
                }
            },
            deep: true
        },

        thinking: {
            handler(val) {
                if (val && Object.keys(val).length > 0) {
                    this.addThinkingTask(val);
                }
            },
            deep: true
        }
    },

    methods: {
        startLoadingSequence() {
            this.tasks = [
                { type: 'status', text: 'Analyzing your question...', status: 'running' }
            ];

            // Simulate progressive loading
            setTimeout(() => {
                if (this.isLoading) {
                    this.tasks.push({ type: 'thinking', text: 'Determining best approach...', status: 'running', expanded: false });
                }
            }, 300);

            setTimeout(() => {
                if (this.isLoading) {
                    this.tasks.push({ type: 'status', text: 'Selecting models...', status: 'running' });
                }
            }, 600);
        },

        addThinkingTask(thinking) {
            const thinkingText = typeof thinking === 'string' ? thinking : thinking.skill_selection || 'Processing...';

            // Update or add thinking task
            const existingThinking = this.tasks.find(t => t.type === 'thinking');
            if (existingThinking) {
                existingThinking.text = thinkingText;
                existingThinking.status = 'complete';
            } else {
                this.tasks.push({
                    type: 'thinking',
                    text: thinkingText,
                    status: 'complete',
                    expanded: false
                });
            }
        },

        updateWithResult(result) {
            // Clear loading tasks and show completed flow
            this.tasks = [];

            // Add task for skill selection
            this.tasks.push({
                type: 'task',
                text: `Select skill: ${result.skill_name}`,
                status: 'complete'
            });

            // Add thinking if available
            if (this.thinking && (typeof this.thinking === 'string' || this.thinking.skill_selection)) {
                const thinkingText = typeof this.thinking === 'string' ? this.thinking : this.thinking.skill_selection;
                this.tasks.push({
                    type: 'thinking',
                    text: thinkingText,
                    status: 'complete',
                    expanded: false
                });
            }

            // Add read data action
            this.tasks.push({
                type: 'read',
                text: 'time series data',
                status: 'complete'
            });

            // Add model executions
            if (result.models_used && result.models_used.length > 0) {
                result.models_used.forEach(model => {
                    this.tasks.push({
                        type: 'execute',
                        text: model,
                        status: 'complete'
                    });
                });
            }

            // Add completion status
            this.tasks.push({
                type: 'status',
                text: result.success ? 'Analysis complete' : 'Analysis failed',
                status: result.success ? 'complete' : 'error'
            });
        }
    }
};

if (typeof window !== 'undefined') {
    window.ThinkingView = ThinkingView;
}
