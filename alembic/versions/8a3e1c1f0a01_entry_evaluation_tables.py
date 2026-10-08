"""entry evaluation mode tables (programs, entries, ocr, cleaning, batches, samples, drive files)

Revision ID: 8a3e1c1f0a01
Revises: e6b3ecaf0683
Create Date: 2026-10-08 11:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8a3e1c1f0a01'
down_revision: Union[str, None] = 'e6b3ecaf0683'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ----- programs --------------------------------------------------------
    op.create_table(
        'programs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('state', sa.String(length=30), nullable=False),
        sa.Column('rubric_id', sa.Integer(), nullable=True),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        sa.Column('forms_csv_path', sa.String(length=600), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['rubric_id'], ['rubrics.id'], name=op.f('fk_programs_rubric_id_rubrics'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_programs')),
        sa.UniqueConstraint('name', name=op.f('uq_programs_name')),
    )
    op.create_index(op.f('ix_programs_state'), 'programs', ['state'], unique=False)

    # ----- evaluation_batches ---------------------------------------------
    op.create_table(
        'evaluation_batches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('program_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=80), nullable=False),
        sa.Column('state', sa.String(length=30), nullable=False),
        sa.Column('target_size', sa.Integer(), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('completed', sa.Integer(), nullable=False),
        sa.Column('failed', sa.Integer(), nullable=False),
        sa.Column('flagged', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['program_id'], ['programs.id'], name=op.f('fk_evaluation_batches_program_id_programs'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_evaluation_batches')),
    )
    op.create_index(op.f('ix_evaluation_batches_program_id'), 'evaluation_batches', ['program_id'], unique=False)
    op.create_index(op.f('ix_evaluation_batches_state'), 'evaluation_batches', ['state'], unique=False)
    op.create_index('ix_eval_batches_program_state', 'evaluation_batches', ['program_id', 'state'], unique=False)

    # ----- drive_files -----------------------------------------------------
    op.create_table(
        'drive_files',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('drive_file_id', sa.String(length=120), nullable=False),
        sa.Column('name', sa.String(length=400), nullable=False),
        sa.Column('mime', sa.String(length=120), nullable=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=True),
        sa.Column('local_path', sa.String(length=600), nullable=True),
        sa.Column('download_url', sa.String(length=800), nullable=True),
        sa.Column('parent_folder', sa.String(length=120), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_drive_files')),
        sa.UniqueConstraint('drive_file_id', name=op.f('uq_drive_files_drive_file_id')),
    )
    op.create_index(op.f('ix_drive_files_drive_file_id'), 'drive_files', ['drive_file_id'], unique=False)
    op.create_index(op.f('ix_drive_files_sha256'), 'drive_files', ['sha256'], unique=False)
    op.create_index(op.f('ix_drive_files_parent_folder'), 'drive_files', ['parent_folder'], unique=False)

    # ----- entries ---------------------------------------------------------
    op.create_table(
        'entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('program_id', sa.Integer(), nullable=False),
        sa.Column('external_ref', sa.String(length=120), nullable=False),
        sa.Column('state', sa.String(length=30), nullable=False),
        sa.Column('form_kind', sa.String(length=40), nullable=False),
        sa.Column('full_name', sa.String(length=200), nullable=True),
        sa.Column('gender', sa.String(length=20), nullable=True),
        sa.Column('phone', sa.String(length=40), nullable=True),
        sa.Column('school_name', sa.String(length=300), nullable=True),
        sa.Column('class_name', sa.String(length=40), nullable=True),
        sa.Column('state_name', sa.String(length=80), nullable=True),
        sa.Column('district', sa.String(length=120), nullable=True),
        sa.Column('block_name', sa.String(length=120), nullable=True),
        sa.Column('division', sa.String(length=120), nullable=True),
        sa.Column('project_office', sa.String(length=120), nullable=True),
        sa.Column('ashramshala', sa.String(length=200), nullable=True),
        sa.Column('language', sa.String(length=40), nullable=True),
        sa.Column('theme', sa.String(length=80), nullable=True),
        sa.Column('ai_description', sa.Text(), nullable=True),
        sa.Column('resources', sa.JSON(), nullable=True),
        sa.Column('situation', sa.Text(), nullable=True),
        sa.Column('proposed_solution', sa.Text(), nullable=True),
        sa.Column('declaration', sa.Boolean(), nullable=False),
        sa.Column('consent', sa.Boolean(), nullable=False),
        sa.Column('po_name', sa.String(length=200), nullable=True),
        sa.Column('school_name_form', sa.String(length=300), nullable=True),
        sa.Column('submission_count', sa.Integer(), nullable=True),
        sa.Column('email', sa.String(length=200), nullable=True),
        sa.Column('submitted_text', sa.Text(), nullable=True),
        sa.Column('total_score', sa.Float(), nullable=True),
        sa.Column('max_score', sa.Float(), nullable=True),
        sa.Column('percentage', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('drive_url', sa.String(length=800), nullable=True),
        sa.Column('drive_file_id', sa.String(length=120), nullable=True),
        sa.Column('local_file_path', sa.String(length=600), nullable=True),
        sa.Column('drive_sha256', sa.String(length=64), nullable=True),
        sa.Column('needs_review', sa.Boolean(), nullable=False),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('review_status', sa.String(length=20), nullable=False),
        sa.Column('human_score', sa.Float(), nullable=True),
        sa.Column('human_score_notes', sa.Text(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['batch_id'], ['evaluation_batches.id'], name=op.f('fk_entries_batch_id_evaluation_batches'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['program_id'], ['programs.id'], name=op.f('fk_entries_program_id_programs'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_entries')),
        sa.UniqueConstraint('program_id', 'external_ref', name=op.f('uq_entries_program_id_external_ref')),
    )
    op.create_index(op.f('ix_entries_program_id'), 'entries', ['program_id'], unique=False)
    op.create_index(op.f('ix_entries_external_ref'), 'entries', ['external_ref'], unique=False)
    op.create_index(op.f('ix_entries_state'), 'entries', ['state'], unique=False)
    op.create_index(op.f('ix_entries_theme'), 'entries', ['theme'], unique=False)
    op.create_index(op.f('ix_entries_drive_file_id'), 'entries', ['drive_file_id'], unique=False)
    op.create_index(op.f('ix_entries_drive_sha256'), 'entries', ['drive_sha256'], unique=False)
    op.create_index(op.f('ix_entries_batch_id'), 'entries', ['batch_id'], unique=False)
    op.create_index('ix_entries_program_state', 'entries', ['program_id', 'state'], unique=False)

    # ----- entry_ocr -------------------------------------------------------
    op.create_table(
        'entry_ocr',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('method', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('language', sa.String(length=20), nullable=True),
        sa.Column('original_text', sa.Text(), nullable=True),
        sa.Column('reconstructed_text', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(length=60), nullable=True),
        sa.Column('model', sa.String(length=120), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('is_selected', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['entry_id'], ['entries.id'], name=op.f('fk_entry_ocr_entry_id_entries'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_entry_ocr')),
    )
    op.create_index(op.f('ix_entry_ocr_entry_id'), 'entry_ocr', ['entry_id'], unique=False)
    op.create_index('ix_entry_ocr_entry', 'entry_ocr', ['entry_id'], unique=False)

    # ----- cleaning_runs ---------------------------------------------------
    op.create_table(
        'cleaning_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('method', sa.String(length=40), nullable=False),
        sa.Column('original_text', sa.Text(), nullable=False),
        sa.Column('cleaned_text', sa.Text(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('provider', sa.String(length=60), nullable=True),
        sa.Column('model', sa.String(length=120), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('applied', sa.Boolean(), nullable=False),
        sa.Column('applied_by', sa.String(length=40), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['entry_id'], ['entries.id'], name=op.f('fk_cleaning_runs_entry_id_entries'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cleaning_runs')),
    )
    op.create_index(op.f('ix_cleaning_runs_entry_id'), 'cleaning_runs', ['entry_id'], unique=False)

    # ----- entry_scores + entry_score_criteria ----------------------------
    op.create_table(
        'entry_scores',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('strategy', sa.String(length=60), nullable=False),
        sa.Column('provider', sa.String(length=60), nullable=True),
        sa.Column('model', sa.String(length=120), nullable=True),
        sa.Column('rubric_id', sa.Integer(), nullable=True),
        sa.Column('total_awarded', sa.Float(), nullable=True),
        sa.Column('total_max', sa.Float(), nullable=True),
        sa.Column('percentage', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('strengths', sa.Text(), nullable=True),
        sa.Column('gaps', sa.Text(), nullable=True),
        sa.Column('observations', sa.Text(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('is_current', sa.Boolean(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['batch_id'], ['evaluation_batches.id'], name=op.f('fk_entry_scores_batch_id_evaluation_batches'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['entry_id'], ['entries.id'], name=op.f('fk_entry_scores_entry_id_entries'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['rubric_id'], ['rubrics.id'], name=op.f('fk_entry_scores_rubric_id_rubrics'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_entry_scores')),
    )
    op.create_index(op.f('ix_entry_scores_entry_id'), 'entry_scores', ['entry_id'], unique=False)
    op.create_index(op.f('ix_entry_scores_batch_id'), 'entry_scores', ['batch_id'], unique=False)
    op.create_index('ix_entry_scores_entry_current', 'entry_scores', ['entry_id', 'is_current'], unique=False)

    op.create_table(
        'entry_score_criteria',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('score_id', sa.Integer(), nullable=False),
        sa.Column('criterion', sa.String(length=80), nullable=False),
        sa.Column('awarded', sa.Float(), nullable=False),
        sa.Column('max_score', sa.Float(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['score_id'], ['entry_scores.id'], name=op.f('fk_entry_score_criteria_score_id_entry_scores'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_entry_score_criteria')),
        sa.UniqueConstraint('score_id', 'criterion', name=op.f('uq_entry_score_criteria_score_id_criterion')),
    )
    op.create_index(op.f('ix_entry_score_criteria_score_id'), 'entry_score_criteria', ['score_id'], unique=False)

    # ----- review_samples --------------------------------------------------
    op.create_table(
        'review_samples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=False),
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('kind', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('reviewer', sa.String(length=120), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('final_score', sa.Float(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['batch_id'], ['evaluation_batches.id'], name=op.f('fk_review_samples_batch_id_evaluation_batches'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['entry_id'], ['entries.id'], name=op.f('fk_review_samples_entry_id_entries'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_review_samples')),
        sa.UniqueConstraint('batch_id', 'entry_id', name=op.f('uq_review_samples_batch_id_entry_id')),
    )
    op.create_index(op.f('ix_review_samples_batch_id'), 'review_samples', ['batch_id'], unique=False)
    op.create_index(op.f('ix_review_samples_entry_id'), 'review_samples', ['entry_id'], unique=False)


def downgrade() -> None:
    op.drop_table('review_samples')
    op.drop_table('entry_score_criteria')
    op.drop_table('entry_scores')
    op.drop_table('cleaning_runs')
    op.drop_table('entry_ocr')
    op.drop_table('entries')
    op.drop_table('drive_files')
    op.drop_table('evaluation_batches')
    op.drop_table('programs')