/**
 * ChatInterface Component - Main chat experience
 */

const ChatInterface = {
    name: 'ChatInterface',

    props: {
        session: {
            type: Object,
            required: true
        }
    },

    emits: ['new-session'],

    components: {
        ThinkingView: window.ThinkingView,
        ModeSelector: window.ModeSelector,
        CouncilView: window.CouncilView,
        TimeSeriesChart: window.TimeSeriesChart
    },

    template: `
        <div class="chat-interface">
            <!-- Header Bar -->
            <div class="chat-header">
                <div class="session-info">
                    <span class="provider-badge">{{ session.provider?.toUpperCase() }}</span>
                    <span class="session-file">{{ session.file_name || session.csv_path }}</span>
                </div>
                <div class="header-actions">
                    <ModeSelector v-model="mode" :disabled="isLoading" />
                    <button class="btn-new-session" @click="$emit('new-session')">
                        New Session
                    </button>
                </div>
            </div>

            <!-- Messages Container -->
            <div class="messages-container" ref="messagesContainer">
                <!-- Welcome Message -->
                <div class="message system">
                    <div class="message-content">
                        <strong>Session started!</strong><br>
                        Analyzing: <strong>{{ session.file_name || session.csv_path }}</strong><br>
                        Target column: <strong>{{ session.target_col }}</strong>
                    </div>
                    
                    <!-- Initial Time Series Chart -->
                    <TimeSeriesChart
                        v-if="historicalChartData && historicalChartData.values"
                        chart-type="overview"
                        :skill-result="{ data: historicalChartData }"
                        :historical-data="historicalChartData"
                        :title="'📊 ' + session.target_col + ' Time Series'"
                        :session-id="session.session_id"
                    />
                    
                    <div class="message-content example-questions-box">
                        <div class="example-questions">
                            <p>Try asking:</p>
                            <ul>
                                <li @click="sendQuickMessage('What will happen next week?')">What will happen next week?</li>
                                <li @click="sendQuickMessage('Are there any anomalies?')">Are there any anomalies?</li>
                                <li @click="sendQuickMessage('Describe the trend')">Describe the trend</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Message History -->
                <template v-for="(msg, index) in messages" :key="index">
                    <!-- User Message -->
                    <div v-if="msg.role === 'user'" class="message user">
                        <div class="user-message-wrapper">
                            <div class="message-content">{{ msg.content }}</div>
                            <!-- Retry Button (shown only if this is the last user message and an error occurred) -->
                            <div v-if="index === messages.length - 2 && lastMessageHadError && !isLoading" class="retry-container">
                                <button class="btn-retry" @click="retryLastMessage" title="Retry this message">
                                    <span class="retry-icon">↺</span> Retry
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Assistant Message -->
                    <div v-else-if="msg.role === 'assistant'" class="message assistant">
                        <!-- Thinking View (Collapsible) -->
                        <ThinkingView
                            v-if="msg.thinking || msg.skill_result"
                            :thinking="msg.thinking || {}"
                            :skill-result="msg.skill_result"
                        />

                        <!-- Main Response -->
                        <div class="message-content" v-html="formatContent(msg.content)"></div>

                        <!-- Time Series Chart (if plottable data) -->
                        <TimeSeriesChart
                            v-if="hasPlottableData(msg.skill_result)"
                            :chart-type="getChartType(msg.skill_result)"
                            :skill-result="msg.skill_result"
                            :historical-data="historicalChartData"
                            :session-id="session.session_id"
                        />

                        <!-- Karpathy-style Council Deliberation (if available) -->
                        <CouncilView
                            v-if="msg.deliberation"
                            :deliberation="msg.deliberation"
                            :perspectives="msg.perspectives"
                        />

                        <!-- Legacy Council Perspectives (fallback) -->
                        <div v-else-if="msg.perspectives" class="council-perspectives">
                            <div
                                v-for="(analysis, role) in msg.perspectives"
                                :key="role"
                                class="perspective"
                                :class="'perspective-' + role"
                            >
                                <div class="perspective-header">{{ formatRoleName(role) }}</div>
                                <div class="perspective-content">{{ analysis }}</div>
                            </div>
                        </div>

                        <!-- Dynamic Suggestions -->
                        <div v-if="msg.suggestions && msg.suggestions.length > 0" class="message-content example-questions-box" style="margin-top: 8px">
                            <div class="example-questions">
                                <p style="margin-bottom: 6px; opacity: 0.7;">Suggested actions:</p>
                                <ul>
                                    <li v-for="sugg in msg.suggestions" :key="sugg" @click="sendQuickMessage(sugg)">{{ sugg }}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </template>

                <!-- Loading Indicator -->
                <div v-if="isLoading" class="message assistant loading">
                    <ThinkingView :is-loading="true" />
                </div>
            </div>

            <!-- Progress Bar -->
            <div v-if="isLoading && progress" class="progress-bar-container">
                <div class="progress-bar">
                    <div class="progress-fill" :style="{ width: progress.percent + '%' }"></div>
                </div>
                <div class="progress-text">{{ progress.message }}</div>
            </div>

            <!-- Input Area -->
            <div class="input-area">
                <div class="context-toggle" v-if="userContext">
                    <span class="context-indicator" title="Context attached">📝</span>
                </div>
                <input
                    ref="messageInput"
                    v-model="inputMessage"
                    type="text"
                    :placeholder="inputPlaceholder"
                    :disabled="isLoading"
                    @keypress.enter="sendMessage"
                >
                <button
                    class="btn-send"
                    :disabled="!inputMessage.trim() || isLoading"
                    @click="sendMessage"
                >
                    {{ isLoading ? '...' : 'Send' }}
                </button>
            </div>
        </div>
    `,

    data() {
        return {
            messages: [],
            inputMessage: '',
            isLoading: false,
            progress: null,
            mode: 'council',
            userContext: '',
            eventSource: null,
            historicalChartData: null,
            lastUserMessage: null,
            lastMessageHadError: false
        };
    },

    computed: {
        inputPlaceholder() {
            if (this.mode === 'multi_model') {
                return 'Ask about forecasts or anomalies (uses multiple AI models)...';
            } else if (this.mode === 'council') {
                return 'Ask about your data (council of AI experts will respond)...';
            }
            return 'Ask about your time series data...';
        }
    },

    mounted() {
        this.mode = this.session.mode || 'council';
        this.userContext = this.session.userContext || '';
        this.loadChatHistory(); // Load chat history on mount
        this.$refs.messageInput?.focus();
        this.fetchHistoricalData();
    },

    beforeUnmount() {
        if (this.eventSource) {
            this.eventSource.close();
        }
    },

    methods: {
        async loadChatHistory() {
            if (!this.session?.session_id) return;
            
            try {
                // Try localStorage first (instant)
                const localKey = `chat_${this.session.session_id}`;
                const localData = localStorage.getItem(localKey);
                if (localData) {
                    this.messages = JSON.parse(localData);
                    console.log('Loaded chat history from localStorage:', this.messages.length, 'messages');
                }
                
                // Then fetch from server (authoritative source)
                const response = await fetch(`/api/session/${this.session.session_id}/history`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.messages && data.messages.length > 0) {
                        // Convert server format to UI format
                        this.messages = data.messages.map(msg => ({
                            role: msg.role,
                            content: msg.content,
                            thinking: msg.metadata?.thinking,
                            skill_result: msg.metadata?.skill_result,
                            suggestions: msg.metadata?.suggestions,
                            // Add other metadata as needed
                        }));
                        console.log('Loaded chat history from server:', this.messages.length, 'messages');
                        
                        // Update localStorage with server data
                        this.saveChatToLocalStorage();
                    }
                }
            } catch (e) {
                console.warn('Could not load chat history:', e);
            }
            
            this.scrollToBottom();
        },
        
        saveChatToLocalStorage() {
            if (!this.session?.session_id) return;
            
            try {
                const key = `chat_${this.session.session_id}`;
                localStorage.setItem(key, JSON.stringify(this.messages));
            } catch (e) {
                console.warn('Could not save to localStorage:', e);
            }
        },
        
        sendQuickMessage(message) {
            this.inputMessage = message;
            this.sendMessage();
        },

        async sendMessage() {
            const message = this.inputMessage.trim();
            if (!message || this.isLoading) return;

            this.inputMessage = '';
            this.lastUserMessage = message; // Track last user message
            this.lastMessageHadError = false; // Reset error flag
            this.messages.push({ role: 'user', content: message });
            this.scrollToBottom();

            this.isLoading = true;
            this.progress = { message: 'Analyzing...', percent: 10 };

            // Connect SSE for progress
            this.connectProgressSSE();

            try {
                const response = await fetch(`/api/chat/${this.session.session_id}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        mode: this.mode,
                        user_context: this.userContext || null
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to get response');
                }

                const data = await response.json();

                // Store skill result for download functionality
                if (data.skill_result) {
                    this.storeSkillResult(data.skill_result);
                }

                this.messages.push({
                    role: 'assistant',
                    content: data.response,
                    thinking: data.thinking ? { skill_selection: data.thinking } : null,
                    skill_result: data.skill_result || data.skill_execution,
                    perspectives: data.perspectives || data.council_perspectives,
                    deliberation: data.deliberation,
                    suggestions: data.suggestions
                });

            } catch (error) {
                this.lastMessageHadError = true; // Mark that an error occurred
                this.messages.push({
                    role: 'assistant',
                    content: `Error: ${error.message}`
                });
            } finally {
                this.isLoading = false;
                this.progress = null;
                this.closeProgressSSE();
                this.saveChatToLocalStorage(); // Save after each exchange
                this.scrollToBottom();
            }
        },

        retryLastMessage() {
            if (!this.lastUserMessage || this.isLoading) return;
            
            // Remove the last error message from messages array
            if (this.messages.length > 0 && this.messages[this.messages.length - 1].role === 'assistant') {
                this.messages.pop();
            }

            // Remove the last user message to prevent duplication
            if (this.messages.length > 0 && this.messages[this.messages.length - 1].role === 'user') {
                this.messages.pop();
            }
            
            // Resend the last user message
            this.inputMessage = this.lastUserMessage;
            this.sendMessage();
        },

        connectProgressSSE() {
            if (this.eventSource) {
                this.eventSource.close();
            }

            this.eventSource = new EventSource(`/api/progress/${this.session.session_id}`);

            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.progress = {
                        message: data.message,
                        percent: data.progress * 100
                    };

                    if (data.complete) {
                        this.closeProgressSSE();
                    }
                } catch (e) {
                    console.error('SSE parse error:', e);
                }
            };

            this.eventSource.onerror = () => {
                this.closeProgressSSE();
            };
        },

        closeProgressSSE() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const container = this.$refs.messagesContainer;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        formatContent(content) {
            if (!content) return '';

            const safeContent = content
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Use marked.js for proper markdown rendering
            if (typeof marked !== 'undefined') {
                // Configure marked for safe rendering
                marked.setOptions({
                    breaks: true,       // Convert \n to <br>
                    gfm: true,          // GitHub Flavored Markdown
                    headerIds: false,   // Don't add IDs to headers
                    mangle: false       // Don't mangle email links
                });
                
                return marked.parse(safeContent);
            }

            // Fallback to basic formatting if marked is not available
            let formatted = safeContent;

            formatted = formatted
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');

            return formatted;
        },

        formatRoleName(role) {
            const names = {
                'forecaster': '📊 Quantitative Analyst',
                'risk_analyst': '⚠️ Risk Analyst',
                'business_explainer': '💼 Business Insights'
            };
            return names[role] || role;
        },

        hasPlottableData(skillResult) {
            // If we don't have a skill result at all, no chart
            if (!skillResult) return false;
            
            const data = skillResult.data || skillResult;
            
            // Check for forecast data
            if (data.predictions || data.forecast) return true;
            
            // Check for anomaly data (both old and new formats)
            if (data.anomaly_indices && data.anomaly_indices.length > 0) return true;
            if (data.anomalies && data.anomalies.length > 0) return true;
            if (data.high_confidence_anomalies && data.high_confidence_anomalies.length > 0) return true;
            
            // Check for multi-model data (check both locations)
            const modelResults = skillResult.model_results || data.model_results;
            if (modelResults && Object.keys(modelResults).length > 0) return true;
            
            // Check for backtest data
            if (data.context_data && data.predictions) return true;
            
            // Check for aggregated data
            if (skillResult.aggregated?.mean_prediction) return true;
            
            // For analysis skills (describe_series, trend, etc.), show chart if we have historical data
            // These skills don't produce forecast data but the chart contextualizes the analysis
            if (this.historicalChartData && this.historicalChartData.values) {
                const analysisSkills = [
                    'describe_series', 'analyze_trend', 'analysis', 'summary',
                    'statistics', 'describe', 'trend', 'seasonality', 'pattern'
                ];
                const skillName = (skillResult.skill_name || '').toLowerCase();
                
                // Show chart for any analysis-type skill
                if (analysisSkills.some(s => skillName.includes(s))) {
                    return true;
                }
                
                // Also show if skill completed successfully and we have meaningful chart data
                // But exclude stats-only skills that don't need charts
                const statsOnlySkills = [
                    'compare_periods', 'compare_series', 'comparison', 
                    'column_stats', 'correlation'
                ];
                
                if (statsOnlySkills.some(s => skillName.includes(s))) {
                    return false;  // Don't show chart for stats-only results
                }
                
                if (skillResult.success && skillResult.data) {
                    // Check if there's actual chart-worthy data
                    if (data.chart_series || data.trend_data || data.decomposition) {
                        return true;
                    }
                }
            }
            
            return false;
        },

        getChartType(skillResult) {
            if (!skillResult) return 'overview';
            const data = skillResult.data || skillResult;
            
            // Multi-model check first - check both locations
            const modelResults = skillResult.model_results || data.model_results;
            if (modelResults && Object.keys(modelResults).length > 1) {
                return 'multi_model';
            }
            
            // Ensemble check
            if (skillResult.aggregated?.mean_prediction) {
                return 'ensemble';
            }
            
            // Anomaly check (both old and new formats)
            if (data.anomaly_indices && data.anomaly_indices.length > 0) {
                return 'anomaly';
            }
            if (data.anomalies && data.anomalies.length > 0) {
                return 'anomaly';
            }
            if (data.high_confidence_anomalies && data.high_confidence_anomalies.length > 0) {
                return 'anomaly';
            }
            
            // Backtest check (must come BEFORE forecast check since backtest has context_data)
            if (data.context_data && data.predictions) {
                return 'backtest';
            }
            
            // Forecast check
            if (data.predictions || data.forecast) {
                return 'forecast';
            }
            
            // Simulation check
            if (data.chart_data && data.chart_data.simulated_values) {
                return 'simulation';
            }
            
            // Default to overview for analysis skills
            return 'overview';
        },

        async fetchHistoricalData() {
            if (!this.session?.session_id) return;
            
            try {
                const response = await fetch(`/api/session/${this.session.session_id}/chart-data`);
                if (response.ok) {
                    this.historicalChartData = await response.json();
                    console.log('Loaded chart context:', this.historicalChartData?.displayed_points, 'points');
                }
            } catch (e) {
                console.warn('Could not load historical chart data:', e);
            }
        },

        async storeSkillResult(skillResult) {
            if (!this.session?.session_id || !skillResult) return;
            
            try {
                await fetch(`/api/session/${this.session.session_id}/store-result`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skill_result: skillResult })
                });
            } catch (e) {
                console.warn('Could not store skill result for download:', e);
            }
        }
    }
};

if (typeof window !== 'undefined') {
    window.ChatInterface = ChatInterface;
}
