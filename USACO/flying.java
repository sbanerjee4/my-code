import java.io.*;
import java.util.*;
import java.lang.*;
public class flying
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("flying.in"));
      int n = in.nextInt();
      in.close();
      boolean[] visited = new boolean[1000000];
      int moves = 0;
      Queue<Integer> q1 = new ArrayDeque<>();
      Queue<Integer> q2 = new ArrayDeque<>();
      q1.add(1);
      q2.add(0);
      visited[1] = true;
      
      while(!q1.isEmpty())
      {
         int num = q1.peek();
         q1.remove();
         moves = q2.peek();
         q2.remove();
         if(num == n)
         {
            System.out.println(moves);
            break;
         }
         if(num > n + 2 || num < 1 || (visited[num] && num != 1)) 
            continue;
         
         visited[num] = true;
         q1.add(num * 3);
         q2.add(moves + 1);
         q1.add(num - 1);
         q2.add(moves + 1);
         q1.add(num + 1);
         q2.add(moves + 1);
      }
   }
}