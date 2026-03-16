/**
 * CouncilView Component - Karpathy-style Multi-LLM Council Display
 * Shows the complete 3-stage deliberation process with multiple LLM providers
 */

const CouncilView = {
    name: 'CouncilView',

    props: {
        deliberation: {
            type: Object,
            default: null
        },
        perspectives: {
            type: Array,
            default: () => []
        },
        isLoading: {
            type: Boolean,
            default: false
        }
    },

    template: `
        <div class="council-view" v-if="hasContent">
            <!-- Council Header -->
            <div class="council-header" @click="isExpanded = !isExpanded">
                <span class="council-icon">&#x1F3DB;</span>
                <span class="council-title">{{ councilTitle }}</span>
                <span class="council-badge" v-if="memberCount">{{ memberCount }} {{ isMultiLLM ? 'LLMs' : 'experts' }}</span>
                <span class="expand-toggle">{{ isExpanded ? '&#x25BC;' : '&#x25B6;' }}</span>
            </div>

            <!-- Deliberation Content -->
            <div v-show="isExpanded" class="council-content">

                <!-- Multi-LLM Council Members -->
                <div v-if="isMultiLLM && members.length > 0" class="members-section">
                    <div class="section-header">
                        <span class="section-icon">&#x1F916;</span>
                        Council Members
                        <span class="chairman-badge" v-if="chairman">Chair: {{ chairman.name }}</span>
                    </div>
                    <div class="members-grid">
                        <div v-for="member in members" :key="member.provider" class="member-chip"
                             :class="{ 'is-chairman': chairman && chairman.name === member.name }">
                            <span class="member-emoji">{{ member.emoji || '&#x1F916;' }}</span>
                            <span class="member-name">{{ member.name }}</span>
                            <span class="member-provider">({{ member.provider }})</span>
                        </div>
                    </div>
                </div>

                <!-- Stage 1: Independent Analyses -->
                <div class="experts-section" v-if="experts.length > 0">
                    <div class="section-header">
                        <span class="section-icon">{{ isMultiLLM ? '&#x31;&#xFE0F;&#x20E3;' : '&#x1F4CA;' }}</span>
                        {{ isMultiLLM ? 'Stage 1: Independent Analyses' : 'Expert Analyses' }}
                    </div>

                    <div
                        v-for="(expert, index) in experts"
                        :key="expert.key || index"
                        class="expert-card"
                        :class="{ 'expanded': expandedExperts[index], 'multi-llm': isMultiLLM }"
                    >
                        <div class="expert-header" @click="toggleExpert(index)">
                            <span class="expert-emoji">{{ expert.emoji || '&#x1F9D1;' }}</span>
                            <div class="expert-info">
                                <span class="expert-name">{{ expert.name || expert.role }}</span>
                                <span class="expert-role">{{ expert.role_title || expert.role }}</span>
                            </div>
                            <span v-if="getExpertRank(expert)" class="expert-rank">#{{ getExpertRank(expert) }}</span>
                            <span class="expert-toggle">{{ expandedExperts[index] ? '&#x2212;' : '+' }}</span>
                        </div>

                        <div v-show="expandedExperts[index]" class="expert-analysis">
                            <div class="analysis-content" v-html="formatMarkdown(expert.analysis)"></div>
                        </div>
                    </div>
                </div>

                <!-- Stage 2: Peer Review Rankings (Multi-LLM only) -->
                <div class="rankings-section" v-if="isMultiLLM && stage2Rankings.length > 0">
                    <div class="section-header" @click="rankingsExpanded = !rankingsExpanded">
                        <span class="section-icon">&#x32;&#xFE0F;&#x20E3;</span>
                        Stage 2: Peer Review Rankings
                        <span class="toggle-icon">{{ rankingsExpanded ? '&#x25BC;' : '&#x25B6;' }}</span>
                    </div>

                    <div v-show="rankingsExpanded" class="rankings-content">
                        <div v-for="(ranking, index) in stage2Rankings" :key="index" class="ranking-item">
                            <div class="ranking-header">
                                <span class="ranker-name">{{ ranking.member }}'s Rankings:</span>
                            </div>
                            <div class="ranking-order">
                                <span v-for="(id, i) in ranking.rankings" :key="id" class="rank-badge">
                                    {{ i + 1 }}. {{ getNameForId(id) || id }}
                                </span>
                            </div>
                            <div class="ranking-reasoning" v-html="formatMarkdown(ranking.reasoning)"></div>
                        </div>

                        <!-- Aggregate Rankings -->
                        <div v-if="aggregateRankings.length > 0" class="aggregate-rankings">
                            <h4>&#x1F3C6; Aggregate Rankings (Borda Count)</h4>
                            <div class="rankings-table">
                                <div v-for="(rank, index) in aggregateRankings" :key="rank.id" class="rank-row"
                                     :class="{ 'top-rank': index === 0 }">
                                    <span class="rank-position">{{ index + 1 }}</span>
                                    <span class="rank-member">{{ rank.member }}</span>
                                    <span class="rank-score">Score: {{ rank.total_score }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Round Table Discussion (Single-LLM) -->
                <div class="roundtable-section" v-if="!isMultiLLM && roundTable.length > 0">
                    <div class="section-header" @click="roundTableExpanded = !roundTableExpanded">
                        <span class="section-icon">&#x1F5E3;</span>
                        Round-Table Discussion
                        <span class="toggle-icon">{{ roundTableExpanded ? '&#x25BC;' : '&#x25B6;' }}</span>
                    </div>

                    <div v-show="roundTableExpanded" class="roundtable-content">
                        <div v-for="(item, index) in roundTable" :key="index" class="discussion-item">
                            <div v-if="item.type === 'panel_discussion'" class="panel-discussion">
                                <div v-html="formatMarkdown(item.content)"></div>
                            </div>
                            <div v-else-if="item.type === 'ranking'" class="ranking-discussion">
                                <strong>{{ item.member }}</strong>
                                <div v-html="formatMarkdown(item.content)"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Stage 3 / Final Synthesis -->
                <div class="synthesis-section" v-if="synthesis">
                    <div class="section-header" @click="synthesisExpanded = !synthesisExpanded">
                        <span class="section-icon">{{ isMultiLLM ? '&#x33;&#xFE0F;&#x20E3;' : '&#x1F9E0;' }}</span>
                        {{ isMultiLLM ? 'Stage 3: Chairman Synthesis' : 'Final Synthesis' }}
                        <span class="synthesis-author" v-if="synthesis.author">by {{ synthesis.author }}</span>
                        <span class="toggle-icon">{{ synthesisExpanded ? '&#x25BC;' : '&#x25B6;' }}</span>
                    </div>

                    <div v-show="synthesisExpanded" class="synthesis-content">
                        <div v-html="formatMarkdown(synthesis.content)"></div>
                    </div>
                </div>

                <!-- Full Transcript Toggle -->
                <div class="transcript-section" v-if="fullTranscript">
                    <button class="transcript-toggle" @click="showTranscript = !showTranscript">
                        {{ showTranscript ? 'Hide' : 'Show' }} Full Transcript
                    </button>

                    <div v-show="showTranscript" class="transcript-content">
                        <pre>{{ fullTranscript }}</pre>
                    </div>
                </div>
            </div>
        </div>
    `,

    data() {
        return {
            isExpanded: true,
            expandedExperts: {},
            rankingsExpanded: false,
            roundTableExpanded: false,
            synthesisExpanded: true,
            showTranscript: false
        };
    },

    computed: {
        hasContent() {
            return this.deliberation || (this.perspectives && this.perspectives.length > 0);
        },

        isMultiLLM() {
            return this.deliberation?.council_type === 'multi_llm';
        },

        councilTitle() {
            return this.isMultiLLM ? 'Multi-LLM Council' : 'Council of AI Experts';
        },

        members() {
            return this.deliberation?.members || [];
        },

        chairman() {
            return this.deliberation?.chairman || null;
        },

        memberCount() {
            if (this.isMultiLLM) {
                return this.deliberation?.member_count || this.members.length;
            }
            return this.experts.length;
        },

        experts() {
            if (this.deliberation && this.deliberation.experts) {
                return this.deliberation.experts;
            }
            // Fallback to perspectives if no deliberation
            return this.perspectives.map(p => ({
                key: p.role,
                name: p.role,
                role: p.role,
                emoji: p.emoji || this.getEmojiForRole(p.role),
                analysis: p.analysis
            }));
        },

        stage2Rankings() {
            return this.deliberation?.stage2 || [];
        },

        aggregateRankings() {
            return this.deliberation?.stage3?.aggregate_rankings || [];
        },

        roundTable() {
            return this.deliberation?.round_table || [];
        },

        synthesis() {
            return this.deliberation?.synthesis || null;
        },

        fullTranscript() {
            return this.deliberation?.full_transcript || null;
        }
    },

    methods: {
        toggleExpert(index) {
            this.expandedExperts = {
                ...this.expandedExperts,
                [index]: !this.expandedExperts[index]
            };
        },

        getExpertRank(expert) {
            if (!this.isMultiLLM || !this.aggregateRankings.length) return null;
            const rankIndex = this.aggregateRankings.findIndex(r => r.member === expert.name);
            return rankIndex >= 0 ? rankIndex + 1 : null;
        },

        getNameForId(anonymousId) {
            // Try to find the member name for an anonymous ID
            const stage1 = this.deliberation?.stage1 || [];
            const match = stage1.find(s => s.anonymous_id === anonymousId);
            return match ? match.member : null;
        },

        getEmojiForRole(role) {
            const emojiMap = {
                'statistician': '&#x1F4CA;',
                'domain_expert': '&#x1F3AF;',
                'risk_analyst': '&#x26A0;',
                'optimist': '&#x1F680;',
                'forecaster': '&#x1F4CA;',
                'business_explainer': '&#x1F4BC;',
                'Claude': '&#x1F7E3;',
                'GPT': '&#x1F7E2;',
                'Gemini': '&#x1F535;',
                'DeepSeek': '&#x1F7E1;',
                'Qwen': '&#x1F7E0;'
            };
            return emojiMap[role] || '&#x1F9D1;';
        },

        formatMarkdown(text) {
            if (!text) return '';

            const safeText = text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Simple markdown-like formatting
            return safeText
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/^### (.*$)/gm, '<h4>$1</h4>')
                .replace(/^## (.*$)/gm, '<h3>$1</h3>')
                .replace(/^# (.*$)/gm, '<h2>$1</h2>')
                .replace(/^- (.*$)/gm, '<li>$1</li>')
                .replace(/\n/g, '<br>');
        }
    },

    mounted() {
        // Expand first expert by default
        if (this.experts.length > 0) {
            this.expandedExperts = { 0: true };
        }
    }
};

if (typeof window !== 'undefined') {
    window.CouncilView = CouncilView;
}
