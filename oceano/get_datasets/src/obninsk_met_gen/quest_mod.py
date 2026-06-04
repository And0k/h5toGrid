"""Enhanced questionary widgets with Ctrl+Enter support and list editing."""

import logging
import sys
from typing import Any, Callable, List, Optional, Tuple, Union

import questionary
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyPressEvent
from questionary import Choice, Separator

logger = logging.getLogger(__name__)

CUSTOM_STYLE = questionary.Style([
    ("qmark", "fg:#673ab7 bold"),
    ("question", "bold"),
    ("answer", "fg:#2196f3 bold"),
    ("pointer", "fg:#673ab7 bold"),
    ("highlighted", "fg:#673ab7 bold"),
    ("selected", "fg:#2196f3"),
    ("instruction", "fg:#858585"),
])


def _clear_lines(n: int) -> None:
    """Clear last N terminal lines."""
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()


def enhanced_select(
    message: str,
    choices: List[Any],
    on_ctrl_enter: Optional[Callable[[Any], Any]] = None,
    show_ctrl_hint: bool = True,
    instruction: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Enhanced select with Ctrl+Enter support - reuses questionary.select.

    Args:
        on_ctrl_enter: Callback to execute on Ctrl+Enter, receives current pointed value
        show_ctrl_hint: Show "[Ctrl+Enter...]" hint in instruction (default True)

    Returns:
        - Normal Enter: selected value
        - Ctrl+Enter: ('ctrl_enter', callback_result) if on_ctrl_enter provided
    """
    # Add Ctrl+Enter hint to instruction only if requested
    if on_ctrl_enter and show_ctrl_hint and not instruction:
        instruction = "[Ctrl+O для быстрого выбора]"

    # Create standard question
    question = questionary.select(message, choices=choices, instruction=instruction, **kwargs)

    # Debug messages (can comment safaly)

    # Add debug key logger to ALL key bindings
    app = question.application
    bindings = app.key_bindings

    # Wrap all existing bindings with debug logging
    original_bindings = list(bindings.bindings)
    logger.debug(f"Total bindings before modifications: {len(original_bindings)}")

    for binding in original_bindings:
        logger.debug(f"Existing binding: {binding.keys}, eager={binding.eager}")

    # Add debug global key logger that fires for ANY key
    @bindings.add(Keys.Any, eager=False)
    def debug_key_logger(event: KeyPressEvent):
        """Log every key press for debugging."""
        key_name = event.key_sequence[0].key if event.key_sequence else "unknown"
        logger.debug(f"   Key pressed: {key_name} (sequence: {event.key_sequence})")

        # Check modifiers
        if event.key_sequence:
            key_obj = event.key_sequence[0]
            logger.debug(f"   Key object: {key_obj}, data: {getattr(key_obj, 'data', 'N/A')}")

    # Add Ctrl+Enter binding if callback provided
    if on_ctrl_enter:
        # Use Ctrl+O as Ctrl+Enter alternative (ControlM is already Enter)
        # Also try other combinations
        @bindings.add(Keys.ControlO, eager=True)  # Ctrl+O
        @bindings.add(Keys.Escape, "enter", eager=True)  # Alt+Enter
        def handler_of_added_key_combination(event):
            logger.debug("Ctrl+O handler triggered!")
            try:
                # Find InquirerControl in layout
                for container in event.app.layout.find_all_controls():
                    if hasattr(container, "get_pointed_at") and hasattr(container, "is_answered"):
                        pointed_value = container.get_pointed_at().value
                        result = on_ctrl_enter(pointed_value)
                        logger.debug(f"  Callback({pointed_value}) returned: {result}")
                        container.is_answered = True
                        event.app.exit(result=("ctrl_enter", result))
                        logger.debug(f"  Exiting with result: ('ctrl_enter', {result})")
                        return
                logger.warning("InquirerControl not found in layout")
            except Exception as e:
                logger.error(f"Ctrl+O error: {e}", exc_info=True)

        @bindings.add(Keys.Escape, "enter", eager=True)
        def alt_enter_handler(event):
            logger.debug("Alt+Enter handler TRIGGERED!")
            handler_of_added_key_combination(event)

        # Try to intercept ControlM (Enter) and check if Ctrl is held
        @bindings.add(Keys.ControlM, eager=True)
        def enter_handler(event):
            logger.debug("ControlM (Enter) handler TRIGGERED!")
            # This will likely conflict with original handler
            # Just log and let original handler process
            logger.debug("  Letting original Enter handler process...")

    logger.debug(f"Total bindings after modifications: {len(bindings.bindings)}")

    return question


def select_then_edit(
    message: str,
    choices: List[str],
    default: str = "",
    erase_intermediate: bool = True,
    new_item_marker_prefix: Optional[str] = None,
    accept_string_immediately: Optional[bool] = None,
    **kwargs,
) -> Optional[str]:
    """
    Show select, then immediately open selected item in text editor.

    Args:
        new_item_marker: If provided and selected item starts with this marker,
                        open text input for new value instead of editing selection.
                        Example: "➕" or "Новый" or "Ввести новое"
        accept_string_immediately: If True (or if None and new_item_marker is not None), return selected value
        without editing when it's a simple string (not a list item header)

    Returns:
        Selected/edited value or None if cancelled
    """
    # Only pass default if it's actually in the choices list
    select_kwargs = kwargs.copy()
    # Make sure default is a string and in the choices
    str_default = str(default) if default is not None else ""
    if str_default and str_default in choices:
        select_kwargs.setdefault("default", str_default)
    selected = questionary.select(message, choices=choices, **select_kwargs).unsafe_ask()
    if selected is None:
        return None

    erase_intermediate and _clear_lines(len(choices) + 2)

    # If this is a "new item" marker, start with empty default
    if new_item_marker_prefix and isinstance(selected, str) and selected.startswith(new_item_marker_prefix):
        edited = questionary.text(f"{message} (новое значение):", default="", **kwargs).unsafe_ask()
        return edited if edited is not None else selected

    # If we should accept strings immediately, return without editing
    if (accept_string_immediately is None and new_item_marker_prefix is not None) or accept_string_immediately:
        return selected

    # Otherwise, allow editing
    edited = questionary.text(f"{message} (редактирование):", default=str(selected), **kwargs).unsafe_ask()

    return edited if edited is not None else selected


def select_and_edit_list(
    message: str,
    list_variants: List[List[str]],
    generated_list: List[str],
    max_preview_items: int = 3,
    max_item_len: int = 60,
    erase_intermediate: bool = True,
) -> Optional[List[str]]:
    """
    Multi-step list editor:
    1. Select list variant (with preview)
    2. Show full list, select item to edit OR accept all
    3. Edit selected item
    4. Return to step 2
    """

    def preview(items: List[str]) -> str:
        """Format list preview."""
        preview_items = "; ".join(
            f"{item[:max_item_len]}..." if len(item) > max_item_len else item
            for item in items[:max_preview_items]
        )
        if len(items) > max_preview_items:
            preview_items += f" ... (+{len(items) - max_preview_items})"
        return preview_items

    # STEP 1: Choose list variant
    choices = [
        Choice(title=f"📋 История {i} ({len(v)} элементов): {preview(v)}", value=("history", v))
        for i, v in enumerate(list_variants, 1)
    ] + [
        Choice(
            title=f"✨ Сгенерированный ({len(generated_list)} элементов): {preview(generated_list)}",
            value=("generated", generated_list),
        ),
        Choice(title="➕ Создать новый список", value=("new", [])),
    ]

    if (selected := questionary.select(message, choices=choices, style=CUSTOM_STYLE).unsafe_ask()) is None:
        return None

    erase_intermediate and _clear_lines(len(choices) + 2)

    mode, current_list = selected
    current_list = list(current_list)  # Copy

    # STEP 2: Edit list items
    while True:
        item_choices = [
            Choice(
                title=f"{i}. {item[:max_item_len]}{'...' if len(item) > max_item_len else ''}",
                value=("edit", i - 1),
            )
            for i, item in enumerate(current_list, 1)
        ] + [
            Separator("─" * 50),
            Choice(title="✅ Принять список [Ctrl+O]", value=("accept", None)),
            Choice(title="➕ Добавить элемент", value=("add", None)),
            Choice(title="🗑️  Удалить элемент", value=("delete", None)),
            Choice(title="↩️  Выбрать другой вариант", value=("back", None)),
        ]

        result = enhanced_select(
            f"{message} ({len(current_list)} элементов):",
            choices=item_choices,
            on_ctrl_enter=lambda x: current_list,  # Return list on Ctrl+Enter
            show_ctrl_hint=False,  # Don't show hint - it's in menu item
            style=CUSTOM_STYLE,
        ).unsafe_ask()

        logger.debug(f"Enhanced select returned: {result} (type: {type(result)})")

        if result is None:
            return None

        # Handle Ctrl+Enter - returns list directly
        if isinstance(result, tuple) and result[0] == "ctrl_enter":
            erase_intermediate and _clear_lines(len(item_choices) + 2)
            return result[1]  # Return the list

        action, data = result

        if action == "accept":
            erase_intermediate and _clear_lines(len(item_choices) + 2)
            return current_list

        elif action == "back":
            erase_intermediate and _clear_lines(len(item_choices) + 2)
            return select_and_edit_list(
                message, list_variants, generated_list, max_preview_items, max_item_len, erase_intermediate
            )

        elif action == "edit":
            erase_intermediate and _clear_lines(len(item_choices) + 2)

            if (
                edited := questionary.text(
                    f"Элемент {data + 1}:", default=current_list[data], style=CUSTOM_STYLE
                ).unsafe_ask()
            ) is not None:
                current_list[data] = edited

            erase_intermediate and _clear_lines(2)

        elif action == "add":
            erase_intermediate and _clear_lines(len(item_choices) + 2)

            (
                new_item := questionary.text("Новый элемент:", style=CUSTOM_STYLE).unsafe_ask()
            ) and current_list.append(new_item)

            erase_intermediate and _clear_lines(2)

        elif action == "delete":
            erase_intermediate and _clear_lines(len(item_choices) + 2)

            if (
                current_list
                and (
                    idx := questionary.select(
                        "Удалить элемент:",
                        choices=[
                            Choice(title=f"{i + 1}. {item[:60]}", value=i)
                            for i, item in enumerate(current_list)
                        ],
                        style=CUSTOM_STYLE,
                    ).unsafe_ask()
                )
                is not None
            ):
                current_list.pop(idx)

            erase_intermediate and _clear_lines((len(current_list) + 1) + 2)


def ask_auto_answer(questionary_func, **kwargs):
    """
    Auto-yes wrapper that mimics questionary UI with Enter key press.

    Args:
        questionary_func: questionary function (text, select, confirm, etc.)
        **kwargs: arguments to pass to questionary function

    Returns:
        Default value as if Enter was pressed
    """
    # Set default style if not provided
    if "style" not in kwargs:
        kwargs["style"] = CUSTOM_STYLE

    # Get the message and default value
    message = kwargs.get("message", "")
    default = kwargs.get("default", "")

    # For select/confirm, get first choice or boolean default
    if questionary_func == questionary.select:
        choices = kwargs.get("choices", [])
        default = choices[0] if choices else ""
    elif questionary_func == questionary.confirm:
        default = kwargs.get("default", False)
        default_str = "Y/n" if default else "y/N"
    elif questionary_func == questionary.checkbox:
        default = ""
    elif questionary_func == questionary.path:
        default = "."

    # Print exactly like Questionary does (using hardcoded ANSI codes)
    if questionary_func == questionary.confirm:
        print(f"\u001b[38;5;98;1m\u001b[0m? {message}\u001b[1m\u001b[0m [{default_str}]\u001b[0m")
    else:
        print(f"\u001b[38;5;98;1m\u001b[0m? {message}\u001b[1m\u001b[0m [{default}]\u001b[0m")
    return default
