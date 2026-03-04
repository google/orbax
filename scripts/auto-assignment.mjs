export default async function autoAssign({ github, context }) {
  console.log('Auto-assignment script started');

  let issueNumber;
  let activeAssigneesList;

  // Hardcoded assignee lists
  const issueAssigneesList = ['selamw1', 'rajasekharporeddy'];

  // Determine event type
  if (context.payload.issue) {
    issueNumber = context.payload.issue.number;
    activeAssigneesList = issueAssigneesList;
    console.log('Event Type: Issue');
  } else {
    console.log('Not an Issue. Exiting.');
    return;
  }

  console.log('Target assignees list:', activeAssigneesList);

  if (!activeAssigneesList || activeAssigneesList.length === 0) {
    console.log('No assignees configured for this type.');
    return;
  }

  // Round-robin assignment
  const selection = issueNumber % activeAssigneesList.length;
  const assigneeToAssign = activeAssigneesList[selection];

  console.log(`Assigning #${issueNumber} to: ${assigneeToAssign}`);

  try {
    await github.rest.issues.addAssignees({
      issue_number: issueNumber,
      owner: context.repo.owner,
      repo: context.repo.repo,
      assignees: [assigneeToAssign],
    });
    console.log('Assignment successful');
  } catch (error) {
    console.log('Failed to assign:', error.message);
  }

  console.log('Auto-assignment script completed');
}