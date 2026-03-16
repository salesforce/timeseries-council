/**
 * Time Series Council - Vue.js Application
 * Main application entry point
 */

const { createApp, ref, computed, onMounted, watch } = Vue;

// Main App Component
const App = {
    components: {
        LandingPage: window.LandingPage,
        ChatInterface: window.ChatInterface
    },

    template: `
        <div id="app" class="app-container">
            <!-- Landing Page -->
            <LandingPage
                v-if="!session"
                @session-started="onSessionStarted"
            />

            <!-- Chat Interface -->
            <ChatInterface
                v-else
                :session="session"
                @new-session="endSession"
            />

            <!-- Footer -->
            <footer class="app-footer">
                <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                    <p>Powered by Multi-Model AI Council • Skills Architecture</p>
                    <button 
                        @click="toggleTheme" 
                        class="btn btn-small"
                        style="display: inline-flex; align-items: center; gap: 6px;"
                        :title="currentTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
                    >
                        <span v-if="currentTheme === 'dark'">☀️</span>
                        <span v-else>🌙</span>
                        {{ currentTheme === 'dark' ? 'Light' : 'Dark' }}
                    </button>
                </div>
            </footer>
        </div>
    `,

    setup() {
        const session = ref(null);
        const currentTheme = ref(localStorage.getItem('theme') || 'light');

        // Check for existing session on mount
        onMounted(async () => {
            // Check server health
            try {
                const response = await fetch('/health');
                const data = await response.json();
                console.log('Server health:', data);
            } catch (e) {
                console.error('Server not reachable:', e);
            }
        });

        const onSessionStarted = (sessionData) => {
            session.value = sessionData;
        };

        const endSession = async () => {
            if (session.value?.session_id) {
                try {
                    await fetch(`/api/session/${session.value.session_id}`, {
                        method: 'DELETE'
                    });
                } catch (e) {
                    console.error('Error ending session:', e);
                }
            }
            session.value = null;
        };

        const toggleTheme = () => {
            const newTheme = currentTheme.value === 'light' ? 'dark' : 'light';
            currentTheme.value = newTheme;
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        };

        return {
            session,
            currentTheme,
            onSessionStarted,
            endSession,
            toggleTheme
        };
    }
};

// Initialize Vue app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = createApp(App);
    app.mount('#app');
});
