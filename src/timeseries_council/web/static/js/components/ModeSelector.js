/**
 * ModeSelector Component - Toggle between analysis modes
 */

const ModeSelector = {
    name: 'ModeSelector',

    props: {
        modelValue: {
            type: String,
            default: 'standard'
        },
        disabled: {
            type: Boolean,
            default: false
        }
    },

    emits: ['update:modelValue'],

    template: `
        <div class="mode-selector" :class="{ 'disabled': disabled }">
            <button
                v-for="option in modes"
                :key="option.value"
                class="mode-btn"
                :class="{ 'active': modelValue === option.value }"
                :disabled="disabled"
                @click="$emit('update:modelValue', option.value)"
            >
                <span class="mode-icon">{{ option.icon }}</span>
                <span class="mode-label">{{ option.label }}</span>
            </button>
        </div>
    `,

    data() {
        return {
            modes: [
                { value: 'standard', label: 'Smart', icon: '🧠' },
                { value: 'council', label: 'Council', icon: '👥' }
            ]
        };
    }
};

if (typeof window !== 'undefined') {
    window.ModeSelector = ModeSelector;
}
